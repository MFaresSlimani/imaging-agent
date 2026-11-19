# imaging-agent: code walkthrough

Nov 19, 2026 · @Fares

This explains every file in `dev\imaging-agent`, in the order to read it, and why each choice was made. Read it with the code open next to it. Functions are named rather than line numbers, because lines move.

## Reading order

| Order | File | What it does |
| --- | --- | --- |
| 1 | `imaging_agent/tools.py` | The four tools and the policy thresholds |
| 2 | `imaging_agent/agent.py` | The planner loop and the guard |
| 3 | `imaging_agent/model.py` | The networks and how uncertainty is computed |
| 4 | `imaging_agent/data.py` | CHASE\_DB1 loading, FOV masks, simulated vessel loss |
| 5 | `scripts/train_chase.py` | Training the real model |
| 6 | `scripts/evaluate_chase.py` | Calibration and the full evaluation |
| 7 | `imaging_agent/synthetic.py` + synthetic scripts | The synthetic sanity check |
| 8 | `tests/test_tools.py` | The five tests |

Start with the tools and the agent, because they are the actual idea. The segmentation model is interchangeable.

## The big picture

The main design rule: **the planner never sees pixels.** Images, probability maps and uncertainty maps stay in a Python `Case` object. Tools read and write that object and return a small JSON dict.

Why: an LLM can't usefully read a 999 x 960 array, it would cost a lot of tokens, and keeping the numbers in Python means the LLM can't invent them. It can only act on what the tools report.

The flow is always: segment baseline, segment follow-up, retry if too uncertain, register, measure both visits, compare, finish.

## tools.py

### POLICY

A plain dict of thresholds at the top of the file. It's a dict so the agent can copy it into every trace, which lets a reader see exactly which rules produced a decision. `evaluate_chase.py` also overwrites some values after calibration.

| Key | Default | On CHASE\_DB1 | Meaning |
| --- | --- | --- | --- |
| `max_uncertainty` | 0.03 | 0.0871 (fitted) | Highest allowed mean std on candidate vessel pixels |
| `max_ambiguous_ratio` | 0.10 | 0.262 (fitted) | Highest allowed share of on-the-fence pixels |
| `min_registration_ncc` | 0.80 | 0.80 (hand-set) | Lowest allowed alignment quality |
| `min_detectable_change` | None | 0.0292 (fitted) | Repeatability limit for calling a change real |
| `max_plausible_change` | 0.30 | 0.30 (hand-set) | Largest believable change between two visits |

The defaults were set from the synthetic runs. On real data, three of them are replaced by calibrated values.

### Case

A dataclass holding everything about one patient pair: `images` (visit name to an H x W x 3 float image in \[0, 1\]), `fov` (a boolean mask of the visible retina), `seg` (per visit: probability map, std map, uncertainty numbers), `shift` (from registration) and `metrics` (from measure).

### \_uncertainty(prob, std, fov)

Returns two numbers.

- **mean\_std:** the average std over "candidate" pixels, where prob > 0.2. Not over the whole image, because most pixels are background where every augmentation agrees on 0. Averaging over those would push the number towards zero and hide the signal. The 0.2 cut-off is low on purpose, so faint vessels the model only half sees still count.
- **ambiguous\_ratio:** pixels with 0.3 < p < 0.7, divided by confident vessel pixels (p > 0.5). It catches a model sitting on the fence even when its augmentations agree with each other. Dividing by the vessel count keeps it independent of how many vessels an eye has. The `max(..., 1)` avoids dividing by zero.

### segment(case, segmenter, visit, tta, mc\_passes)

Calls the segmenter, stores prob and std in the case, and returns the two numbers plus `within_policy` (true when both are under their thresholds). It also returns `n_samples`, so the trace shows how much work each attempt cost.

### register(case)

Uses phase cross-correlation on the **green channel**. In fundus photos vessels have the best contrast in green, red is often saturated and blue is noisy. Phase correlation finds the translation between the two images through the Fourier transform, and `upsample_factor=10` gives a precision of a tenth of a pixel.

It then shifts the follow-up and computes the correlation (NCC) only inside the overlap of both fields of view. Otherwise the black area outside the retina would match perfectly and inflate the score. Its limitation is that it only handles translation, with no rotation or deformation.

### \_vessel\_metrics(mask, region)

- **density:** vessel pixels divided by region pixels.
- **length:** skeleton pixels. `skeletonize` thins every vessel to a one-pixel line, so the count approximates total vessel length.
- **mean\_width:** area divided by length.

Only density is used in decisions. The others are logged for later.

### measure(case, visit)

Moves the follow-up's prob and std maps into baseline coordinates using the registration shift, so both visits are measured on the same piece of retina. The region is the baseline FOV intersected with the shifted FOV.

It computes the metrics three times: on p > 0.5 (the point value), on p - std > 0.5 (pessimistic) and on p + std > 0.5 (optimistic). The last two give the uncertainty band.

### compare(case)

`relative_change` = (follow-up - baseline) / baseline. Two rules decide whether it's real:

- If `min_detectable_change` is set, the change is significant when its absolute value exceeds that limit.
- Otherwise it falls back to the band rule: significant only if the two visits' bands don't overlap.

Both exist because the band rule was the original idea, and it failed on real data (1 of 8 detected), since uncertainty along vessel borders makes the bands very wide. `plausible` is true when the absolute change is at most 0.30. It also builds the summary sentence.

## agent.py

### TOOL\_SPECS

A JSON schema for every tool, in the Anthropic tool-use format. Enums limit the values: `visit` must be baseline or followup, `tta` must be none, basic or full, and `mc_passes` runs from 1 to 4. So the LLM can only request valid actions, and the schema also works as documentation.

`finish` is a tool too, so ending a case is an explicit action that gets logged like any other.

### SYSTEM\_TEMPLATE

The procedure written out in words, with the current policy pasted in as JSON. It's formatted at run time, not when the file is imported, because the thresholds change after calibration. The first version was an f-string frozen at import, which would have shown the LLM the old thresholds. That was fixed.

### `Agent.__init__`

- `planner="auto"` picks Claude if `ANTHROPIC_API_KEY` is set, otherwise the rule planner.
- The model comes from the `AGENT_MODEL` environment variable. Check that the default model name is still valid before running the Claude planner.
- `max_steps=12` is a safety cap so a confused LLM can't loop forever.
- `state` tracks `seg_ok`, `reg_ok` and `plausible`. It also has a `retried` set that's never used: a leftover you can delete.

### call(name, args)

The single entry point for every tool. It runs the tool, updates `state`, and appends a trace entry with the step number, planner, tool, arguments, result and time in ms.

Why one entry point: both planners go through it, so neither can skip the logging or the guard.

### \_guard\_finish(args)

A "report" is refused if any segmentation is outside policy, if registration failed, if the change is implausible, or if not all checks ran (fewer than two segmentations, or no registration). It returns `accepted: False` with the reasons. Claude sees that error and has to choose again. A "defer" is always accepted.

Why: safety lives in code, not in the prompt. A prompt can be ignored, a guard can't.

### \_run\_rules()

The procedure as deterministic code: segment each visit with basic TTA. If it's outside policy, retry once with full TTA and 4 MC passes (8 x 4 = 32 samples). If it's still outside, defer with the numbers in the message. Then register, defer if alignment fails, measure, compare, defer if implausible, otherwise report.

### \_run\_claude()

The standard tool-use loop: send the messages, run every `tool_use` block through `call`, send the results back as `tool_result`, and stop at an accepted `finish`. If the loop runs out of steps, it forces a defer. **This path has never been run**, only written.

## model.py

### SmallUNet

Used only for the synthetic check. It has 4 levels with 16, 32, 64 and 128 channels, and each block is two 3 x 3 convolutions with BatchNorm and ReLU.

`Dropout2d(0.2)` sits only in the deep layers (e3, the bottleneck, d3). Dropout in early layers would wreck the fine edges of thin vessels. Deep layers hold more abstract features, where random sampling makes more sense. `Dropout2d` drops whole channels instead of single values, which suits convolutional features.

### SmpUNet

The real model: a segmentation-models-pytorch U-Net with a ResNet-34 encoder pretrained on ImageNet. It matches your A2I-26 setup, and the pretrained encoder helps a lot with only 16 training images.

Two additions:

- `Dropout2d(0.2)` before the segmentation head. The stock smp U-Net has no dropout, so MC dropout would return identical samples.
- ImageNet mean and std normalisation inside `forward`, stored as buffers. Callers pass plain \[0, 1\] images, and the checkpoint carries its own normalisation.

### \_TTA and TTA\_LEVELS

The 8 dihedral transforms: identity, horizontal flip, vertical flip, both flips, rotations by 90, 180 and 270 degrees, and a flip plus rotation. Each comes as a forward and inverse pair.

Why these: they're exact (no interpolation blur), and a retina still looks like a retina when flipped or rotated. The inverse brings each prediction back to the original orientation before averaging.

`TTA_LEVELS`: none = 1, basic = 2 (identity plus horizontal flip), full = 8. Basic is the cheapest setting that produces any spread at all, and full is the retry.

### Segmenter.predict(img, tta, mc\_passes)

1. Pads the image to a multiple of 32. ResNet-34 halves the resolution 5 times (2^5 = 32), and CHASE images are 999 x 960, which isn't divisible.
2. Switches on dropout only. `model.eval()` keeps BatchNorm frozen, and `_enable_mc_dropout` puts just the dropout modules back into training mode. Switching on BatchNorm too would change the statistics and add noise that isn't uncertainty.
3. Loops over augmentations x passes, applies a sigmoid, undoes the transform.
4. Crops the padding off and returns the mean, the std (population std) and the sample count.

It runs under `torch.no_grad()`, and checkpoints load with `weights_only=True`, which is safer with pickle files.

## data.py

### SPLITS

The split is by child (subject), not by image: train 01-08 (16 images), calibration 09-10 (4), test 11-14 (8). The two eyes of one child look very alike, so putting one in training and the other in testing would leak information and inflate the scores. Subjects 11-14 as the test set follows the split commonly used for CHASE\_DB1 (first 20 images train, last 8 test). The calibration subjects were carved out of the 20 training images.

### load\_chase(root, split)

Loads each image as floats in \[0, 1\], uses the first annotator's masks (`1stHO`), and keeps only the vessels inside the field of view.

### fov\_mask(img)

CHASE\_DB1 ships no field-of-view masks, so they're computed:

1. Red channel > 0.08. The background is black and the retina is bright in red.
2. A binary opening with a disk of radius 5 removes small specks.
3. Holes are filled and only the largest connected region is kept.
4. The mask is eroded by 8 pixels. The rim of the retina is dark and noisy, and removing it avoids false vessels along the edge.

### remove\_thin\_vessels(img, gt, fov, seed, fraction=0.5)

This simulates vessel loss on a real image.

1. **Find thin vessels.** An opening with a radius-2 disk keeps only vessels wider than about 5 pixels ("thick"). Thin = all vessels minus thick, with fragments of 20 pixels or fewer removed.
2. **Pick segments.** Thin vessels are split into connected segments (8-connectivity), and a random half is chosen.
3. **Region to erase.** The chosen segments, dilated by 3 pixels to cover the blurred vessel border.
4. **Fill.** Normalised convolution: the image blurred using only vessel-free background pixels, divided by the blurred background mask (sigma 6). It gives the local retina colour while ignoring vessels.
5. **Texture.** Fine detail (image minus a sigma-2 blur), borrowed from 25 pixels to the side and added at half strength, so the patch isn't suspiciously smooth.
6. **Blend.** A soft alpha mask (the region blurred with sigma 1) avoids hard edges.

It returns the new image and the new ground truth (old vessels minus the erased region). The first version used biharmonic inpainting, but erased vessels stayed half visible (green 0.32 against a background of 0.36). The current method gives 0.341 against 0.358, and removes 6 to 10% of the vessel area.

## train\_chase.py

- **Crops of 384 x 384.** The 4 GB GPU can't fit full images in batches, and random crops also give more varied samples from only 16 images.
- **Augmentation:** random 90-degree rotations, horizontal flips, brightness x 0.85 to 1.15 and an offset of plus or minus 0.06, to imitate exposure differences between cameras and visits.
- **Loss:** binary cross-entropy plus soft Dice. Vessels cover only about 10% of the pixels, so BCE alone tends to favour background. Dice targets overlap directly.
- **Optimiser:** AdamW (lr 3e-4, weight decay 1e-4) with a OneCycle schedule peaking at 1e-3, over 3000 steps with batch size 6.
- **Mixed precision** (autocast with GradScaler) saves memory and time on the 3050.
- **Seed 0** for torch and NumPy, so runs are repeatable.
- **Evaluation:** test metrics inside the FOV at threshold 0.5, with no TTA. The test set was only touched once, after training, with no early stopping on it.

## evaluate\_chase.py

### Settings

- `STRENGTHS = [0, 0.4, 0.7, 1.0]`: no, mild, medium and strong degradation.
- `BAD_DICE = 0.70`: below this, a segmentation counts as unusable. It's a reasonable choice, not a derived one, so be ready to say that.

### Part 1: uncertainty against error

`score_images` segments every image at every strength with basic TTA, and records Dice and both uncertainty numbers. The calibration and test images use different random seeds (100+ and 200+).

`best_threshold` uses **Youden's J**: it picks the threshold that maximises true positive rate minus false positive rate. It runs on calibration images only, separately for mean std and for ambiguous ratio. A test image is flagged when either number is over its threshold. `auroc` is a simple pairwise AUROC, and it came out at 1.0.

### Part 1b: repeatability limit

Each calibration image gets a stable follow-up, and the agent's tools measure the density drift between the two. The limit is 1.5 x the worst drift: 1.5 x 0.0195 = 0.0292. The 1.5 is a safety margin because there are only 4 samples, and it's another hand-picked number.

### make\_followup(s, scenario, seed)

- **stable:** exposure x 1.05 + 0.01.
- **progression:** `remove_thin_vessels`.
- **poor\_quality:** progression plus `degrade` at strength 1.

All three get a random camera shift of up to 15 pixels. It returns the ground truth **before** the shift, so the true change is computed correctly. Computing it after the shift was a bug that made stable cases look like they had changed. It was found and fixed.

### Part 2: the agent

The 8 test eyes x 3 scenarios run through the rule planner. `correct()` defines success: stable must be reported as not significant, progression as a significant decrease, and poor quality must be deferred.

The 0.30 plausibility limit isn't fitted anywhere. It was set by hand after the 14R failure on the test set.

## synthetic.py and the synthetic scripts

- **make\_tree:** grows vessels as random walks from the optic disc. There are 4 to 6 trunks, steps of 1.6 pixels, angle noise of 0.12 rad, and each branch has 0.6 to 0.8 times its parent's radius, down to a depth of 4. Growth stops near the edge of the field of view.
- **render:** draws disks along every vessel to make the mask, then builds the colours from a vignette, a glowing optic disc, smooth noise texture, and vessels darker in every channel (most in green, -0.28), like a real fundus.
- **progress:** drops small branches (depth 3 or more) with probability 0.3 and thins the rest to 0.93 of their radius. That gives about 19 to 23% vessel loss. It was 30% before, which was unrealistically large and hit the plausibility limit.
- **shift** and **degrade:** a camera shift, and haze centred at a random point, with blur scaled to the image size plus noise.
- **make\_case:** builds the three scenarios, like `make_followup` for real data.
- **train\_synthetic.py** trains on seeds 10000 and up and validates on seeds 50000 and up, while the demo uses seeds under 10000, so the demo never runs on training images.
- **run\_demo.py** and **figure\_chase.py** only run cases and draw the figures.

## tests/test\_tools.py

`OracleSegmenter` returns the ground truth as its "prediction" with low std, or a flat 0.5 with high std for visits marked noisy. So the tests check the agent's logic without depending on any trained model. It recognises visits by object identity (`img is self.images[v]`), which is a shortcut, not good practice.

The five tests:

1. Registration recovers the true shift to within 1 pixel.
2. Vessel loss is reported as a decrease.
3. An uncertain follow-up is retried with full TTA, then deferred.
4. The guard refuses a report after a failed segmentation.
5. A confidently wrong segmentation (vessels vanish with low std) is deferred as implausible.

## Every number and where it came from

| Value | Where | Source |
| --- | --- | --- |
| 0.0871 max mean std | POLICY | Fitted on calibration images (Youden's J) |
| 0.262 max ambiguous ratio | POLICY | Fitted on calibration images (Youden's J) |
| 0.0292 repeatability limit | POLICY | 1.5 x worst drift on stable calibration pairs |
| 0.80 min NCC | POLICY | Hand-set |
| 0.30 max plausible change | POLICY | Hand-set, after seeing a test failure |
| 0.2 candidate cut-off | `_uncertainty` | Hand-set |
| 0.3 to 0.7 ambiguous band | `_uncertainty` | Hand-set |
| 0.70 bad Dice | evaluate\_chase | Hand-set |
| 1.5 safety margin | evaluate\_chase | Hand-set |
| 0.2 dropout | model | Common default |
| 384 crop, batch 6 | train\_chase | Limited by 4 GB of VRAM |
| 3000 steps, lr 3e-4 to 1e-3 | train\_chase | Common defaults, not tuned |
| 0.08 red threshold, 8 px erosion | fov\_mask | Hand-set, checked visually |

Only three thresholds are fitted. Say this plainly if asked.

## Known weak spots in the code

- `state["retried"]` is unused.
- The Claude planner has never actually run, and its default model name needs checking.
- `OracleSegmenter` matches images by identity.
- Registration is translation only.
- The 0.30 rule was added after looking at test results.
- In `results_band_rule.json`, the true-change column for stable cases was computed with the old bug. The decisions in that file are still valid.

## Questions you may be asked

- **Why TTA instead of deep ensembles?** It's cheap and needs only one model, which matters on a 4 GB GPU. Ensembles are probably better calibrated, but I didn't compare them, and that's a gap.
- **Why average std only over candidate pixels?** Background pixels always agree, so including them would drown the signal.
- **Why calibrate on separate subjects?** Thresholds fitted on the test set make the test results optimistic.
- **Isn't uncertainty the same as error?** No. Case 14R was wrong and confident at the same time, which is exactly why there's an independent plausibility check.
- **Why a guard if the prompt already says what to do?** LLMs sometimes ignore instructions. A guard in code can't be talked out of its rule.
- **How would this transfer to tumours?** Swap the vessel tools for nnU-Net lesion segmentation, add lesion matching across time points, measure by RECIST, and keep the same uncertainty, plausibility and guard layers.
- **What would you do with more time?** Real image-quality labels, a cross-dataset test on DRIVE, STARE and HRF, risk-coverage curves with confidence intervals, and thresholds fixed before testing.
