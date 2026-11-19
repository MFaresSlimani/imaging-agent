import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_agent import tools  # noqa: E402
from imaging_agent.agent import Agent  # noqa: E402
from imaging_agent.synthetic import make_case  # noqa: E402
from imaging_agent.tools import Case  # noqa: E402


class OracleSegmenter:
    """Returns ground truth as the 'prediction' so tool logic is tested without a model."""

    def __init__(self, gt, noisy=()):
        self.gt, self.noisy = gt, noisy

    def predict(self, img, tta="basic", mc_passes=1):
        for v, m in self.gt.items():
            if img is self.images[v]:
                p = m.astype(np.float32)
                std = np.full_like(p, 0.25 if v in self.noisy else 0.01)
                return (np.full_like(p, 0.5) if v in self.noisy else p), std, 1
        raise KeyError


def _setup(scenario, noisy=()):
    d = make_case(3, scenario)
    case = Case({"baseline": d["baseline"], "followup": d["followup"]}, d["fov"])
    seg = OracleSegmenter(d["gt"], noisy)
    seg.images = case.images
    return d, case, seg


def test_registration_recovers_shift():
    d, case, _ = _setup("stable")
    r = tools.register(case)
    assert np.allclose(r["shift_px"], [-s for s in d["true_shift"]], atol=1.0)
    assert r["within_policy"]


def test_progression_reported_as_decrease():
    _, case, seg = _setup("progression")
    res = Agent(case, seg, planner="rules").run()
    assert res["decision"]["decision"] == "report"
    assert "decrease" in res["decision"]["summary"].lower()


def test_uncertain_followup_is_deferred():
    _, case, seg = _setup("progression", noisy=("followup",))
    res = Agent(case, seg, planner="rules").run()
    assert res["decision"]["decision"] == "defer"
    assert any(t["args"].get("tta") == "full" for t in res["trace"])  # it retried first


def test_guard_refuses_unsafe_report():
    _, case, seg = _setup("progression", noisy=("followup",))
    agent = Agent(case, seg, planner="rules")
    agent.call("segment", {"visit": "followup"})
    r = agent.call("finish", {"decision": "report", "summary": "looks fine"})
    assert r["accepted"] is False


def test_implausible_change_is_deferred():
    """A confidently-wrong segmentation (vessels vanish, low uncertainty) must not be reported."""
    d, case, seg = _setup("progression")
    seg.gt = {"baseline": d["gt"]["baseline"], "followup": d["gt"]["followup"] & (np.random.default_rng(0).random(d["fov"].shape) < 0.3)}
    res = Agent(case, seg, planner="rules").run()
    assert res["decision"]["decision"] == "defer"
    assert "implausible" in res["decision"]["summary"]
