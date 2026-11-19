"""Uncertainty-aware longitudinal imaging agent.

The agent plans with either Claude (tool use) or an offline rule planner.
Both go through the same tools and the same guard, and every step is written
to an audit trace.
"""
from __future__ import annotations

import json
import os
import time

from . import tools
from .tools import POLICY, Case

TOOL_SPECS = [
    {"name": "segment", "description": "Segment retinal vessels for one visit and report predictive uncertainty. "
     "tta: none|basic|full (1, 2 or 8 test-time augmentations). mc_passes: MC-dropout passes per augmentation (1-4).",
     "input_schema": {"type": "object", "properties": {
         "visit": {"type": "string", "enum": ["baseline", "followup"]},
         "tta": {"type": "string", "enum": ["none", "basic", "full"]},
         "mc_passes": {"type": "integer", "minimum": 1, "maximum": 4}}, "required": ["visit"]}},
    {"name": "register", "description": "Align follow-up to baseline. Returns shift and alignment quality (NCC).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "measure", "description": "Vessel density and length for a segmented visit, with an uncertainty band.",
     "input_schema": {"type": "object", "properties": {
         "visit": {"type": "string", "enum": ["baseline", "followup"]}}, "required": ["visit"]}},
    {"name": "compare", "description": "Compare measured visits; says whether the change exceeds uncertainty.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "finish", "description": "End the case. decision=report gives a result; decision=defer hands the case "
     "to a human with a reason. A report is refused while any check is outside policy.",
     "input_schema": {"type": "object", "properties": {
         "decision": {"type": "string", "enum": ["report", "defer"]},
         "summary": {"type": "string"}}, "required": ["decision", "summary"]}},
]

SYSTEM_TEMPLATE = """You assess change in retinal vessels between two visits of the same eye.
Use the tools; never guess numbers. Policy: {policy}.
Procedure: segment both visits; if a segmentation is outside policy, retry it once with tta=full and mc_passes=4;
if it is still outside policy, defer and say why. Register; if alignment is outside policy, defer.
Otherwise measure both visits and compare. If compare says the change is not plausible, defer: the model may be
confidently wrong. Otherwise report. Only claim a change if compare says it is beyond uncertainty.
Keep summaries to two sentences, written for a clinician."""


class Agent:
    def __init__(self, case: Case, segmenter, planner: str = "auto", model: str | None = None, max_steps: int = 12):
        self.case, self.segmenter, self.max_steps = case, segmenter, max_steps
        self.model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
        if planner == "auto":
            planner = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "rules"
        self.planner = planner
        self.trace: list[dict] = []
        self.state = {"retried": set(), "seg_ok": {}, "reg_ok": None}

    # ---- tool execution + guard -------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        t0 = time.time()
        c = self.case
        if name == "segment":
            res = tools.segment(c, self.segmenter, args["visit"], args.get("tta", "basic"), args.get("mc_passes", 1))
            self.state["seg_ok"][args["visit"]] = res["within_policy"]
        elif name == "register":
            res = tools.register(c)
            self.state["reg_ok"] = res["within_policy"]
        elif name == "measure":
            res = tools.measure(c, args["visit"])
        elif name == "compare":
            res = tools.compare(c)
            self.state["plausible"] = res.get("plausible", True)
        elif name == "finish":
            res = self._guard_finish(args)
        else:
            res = {"error": f"unknown tool {name}"}
        self.trace.append({"step": len(self.trace) + 1, "planner": self.planner, "tool": name,
                           "args": args, "result": res, "ms": round((time.time() - t0) * 1000)})
        return res

    def _guard_finish(self, args):
        if args["decision"] == "report":
            problems = [f"{v} segmentation outside policy" for v, ok in self.state["seg_ok"].items() if not ok]
            if self.state["reg_ok"] is False:
                problems.append("registration outside policy")
            if self.state.get("plausible") is False:
                problems.append("measured change is physiologically implausible")
            if len(self.state["seg_ok"]) < 2 or self.state["reg_ok"] is None:
                problems.append("not all checks were run")
            if problems:
                return {"accepted": False, "error": "report refused by guard: " + "; ".join(problems)}
        return {"accepted": True, **args}

    # ---- planners ---------------------------------------------------------------
    def run(self) -> dict:
        result = self._run_claude() if self.planner == "claude" else self._run_rules()
        return {"decision": result, "trace": self.trace, "policy": dict(POLICY), "planner": self.planner,
                "model": self.model if self.planner == "claude" else None}

    def _run_rules(self):
        for v in ("baseline", "followup"):
            r = self.call("segment", {"visit": v, "tta": "basic"})
            if not r["within_policy"]:
                r = self.call("segment", {"visit": v, "tta": "full", "mc_passes": 4})
                if not r["within_policy"]:
                    return self.call("finish", {"decision": "defer", "summary":
                        f"The {v} image is too uncertain to measure (mean std {r['mean_std']}, ambiguous ratio "
                        f"{r['ambiguous_ratio']}) even after 8x TTA and MC dropout. Please review image quality."})
        r = self.call("register", {})
        if not r["within_policy"]:
            return self.call("finish", {"decision": "defer", "summary":
                f"Visits could not be aligned reliably (NCC {r['ncc']}). Please review manually."})
        self.call("measure", {"visit": "baseline"})
        self.call("measure", {"visit": "followup"})
        cmp = self.call("compare", {})
        if not cmp["plausible"]:
            return self.call("finish", {"decision": "defer", "summary":
                f"Measured {cmp['summary']}, which is implausible between two visits. One acquisition likely failed "
                "although the model was confident. Please review both images."})
        return self.call("finish", {"decision": "report", "summary": cmp["summary"].capitalize() + "."})

    def _run_claude(self):
        import anthropic
        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": "New case: baseline and follow-up fundus images are loaded. Assess it."}]
        for _ in range(self.max_steps):
            system = SYSTEM_TEMPLATE.format(policy=json.dumps(POLICY))
            resp = client.messages.create(model=self.model, max_tokens=1024, system=system,
                                          tools=TOOL_SPECS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                break
            results = []
            for u in uses:
                res = self.call(u.name, dict(u.input))
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(res)})
                if u.name == "finish" and res.get("accepted"):
                    return res
            messages.append({"role": "user", "content": results})
        return self.call("finish", {"decision": "defer", "summary": "Agent stopped without a valid decision."})
