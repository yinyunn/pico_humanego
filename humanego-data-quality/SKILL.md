---
name: humanego-data-quality
description: Audit and visualize HumanEgo/PICO-to-HumanEgo training data quality before training. Checks HumanEgo preprocessing outputs, image quality, masks, timestamps, SE(3) poses, hand/object trajectories, duplicate frames, train/eval distribution, and produces an offline HTML report plus CSV/JSON artifacts. Use when asked to inspect dataset quality, diagnose suspicious HumanEgo training/eval behavior, compare demonstrations, find bad episodes/frames, or decide whether data should be cleaned/recollected.
---

# HumanEgo Data Quality

Use this skill to answer one question: **is the HumanEgo training data trustworthy, sufficiently diverse, and internally consistent before blaming the model?**

The official HumanEgo preprocessing output is expected at:

```text
data/<task>/<source>/mps_<task>_<id>_vrs/
└── preprocess/
    └── all_data/<frame>/
        ├── training_data.json
        ├── rgb.png
        ├── rgb_WoArm.png
        ├── mask_arm.png
        ├── mask_obj*.png
        ├── aria_hands.json
        ├── aria_slam.json
        └── ...
```

`training_data.json` is the canonical source for training-facing geometry. Expect at least:

```text
metadata.idx, metadata.ts, metadata.fps, metadata.k, metadata.c2w,
metadata.anchor_key, metadata.is_finished,
entities.hands.<side>.T_hand_to_world, entities.hands.<side>.grasp,
entities.objects.<obj>.T_obj_to_world
```

The tool is intentionally source-agnostic after preprocessing. A PICO recording is supported when it has been converted into the same HumanEgo `preprocess/all_data/<frame>/training_data.json` contract. Raw PICO-only synchronization or skeleton formats require an adapter and must not be silently guessed.

## Safety / data-integrity rules

1. **Never modify, delete, rename, or overwrite the original dataset.**
2. Write all outputs under a separate report directory, default: `quality_reports/<task>/<timestamp>/`.
3. Do not auto-delete suspicious frames. Flag them in `issues.csv` and recommend actions.
4. Do not call a frame "bad" from one weak heuristic alone. Distinguish `info`, `warning`, and `error`.
5. Never fabricate reprojection error. Only compute it when an explicit reprojection adapter/config supplies the 3D point coordinate frame and projection convention.
6. Prefer episode-level diagnosis over random frame deletion; temporal robot demonstrations must preserve action continuity.

## Primary workflow

### 1. Locate the dataset

From the HumanEgo repo root, first identify the task and source(s):

```bash
python humanego-data-quality/scripts/audit_dataset.py \
  --data-root ./data \
  --task <TASK>
```

If the user points at one recording instead:

```bash
python humanego-data-quality/scripts/audit_dataset.py \
  --session ./data/<task>/<source>/<recording>
```

If discovery returns zero sessions, inspect the path. Do not invent a layout.

### 2. Run the audit

Recommended full audit:

```bash
python humanego-data-quality/scripts/audit_dataset.py \
  --data-root ./data \
  --task <TASK> \
  --config humanego-data-quality/quality_config.yaml \
  --output quality_reports/<TASK>/latest
```

For a quick smoke test on a very large dataset:

```bash
python humanego-data-quality/scripts/audit_dataset.py \
  --data-root ./data --task <TASK> \
  --image-stride 10 \
  --output quality_reports/<TASK>/smoke
```

### 3. Read outputs in this order

Always inspect:

1. `report.html` — human-readable dashboard.
2. `summary.json` — machine-readable overall result.
3. `episode_metrics.csv` — rank episodes by suspiciousness.
4. `issues.csv` — exact episode/frame/reason.
5. `frame_metrics.csv` — detailed per-frame metrics.
6. `plots/` — distributions and trajectory views.

### 4. Diagnose by failure family

Use these interpretations:

- **Image problem**: low blur score, extreme brightness, low contrast/entropy, high near-consecutive duplicate ratio.
- **Segmentation problem**: empty masks, implausible arm/object mask coverage, abrupt mask-area jumps.
- **Tracking/geometry problem**: missing hand/object transforms, invalid SE(3), high translation/rotation jumps, discontinuous camera trajectory.
- **Temporal problem**: non-monotonic timestamps, large timestamp gaps, FPS mismatch/jitter.
- **Task-label problem**: grasp stuck at 0/1, no grasp transition where expected, `is_finished` missing, anchor changes inside an episode.
- **Diversity problem**: episodes occupy nearly identical image-feature/trajectory distributions; train/eval distribution is strongly separated.
- **Possible leakage/redundancy**: high exact duplicate hashes shared across episodes/splits.

### 5. Report findings with evidence

When answering the user, do not only say "data quality is bad". Give:

- overall quality score and caveat that it is heuristic;
- top 3–5 failure modes;
- affected episode IDs and frame ranges;
- whether each issue likely affects vision input, ICT/state input, target trajectory, or train/eval generalization;
- concrete action: keep, review, re-preprocess, recalibrate, or recollect.

## What the scripts check

### Schema / completeness

- `training_data.json` readability
- required metadata
- training image path existence
- image dimensions against metadata
- hand/object entity availability
- anchor consistency

### Image quality

Computed on `--image-name` (default `rgb_WoArm_WArmObjKpts.png`, fallback to `rgb.png`):

- brightness mean
- contrast (gray standard deviation)
- Laplacian variance blur score
- grayscale entropy
- under/over-exposed pixel ratio
- exact duplicate dHash
- near-consecutive dHash distance

These are diagnostics, not universal semantic quality measures.

### Segmentation

When masks are available:

- arm mask coverage
- combined/object mask coverage
- empty/nearly-empty mask rate
- abrupt coverage change

### Time / synchronization

From `metadata.ts` and `metadata.fps`:

- timestamp monotonicity
- median `dt`
- effective FPS
- jitter around the median `dt`
- large frame gaps

This validates the **post-preprocessing training timeline**. It does not prove raw RGB↔PICO skeleton synchronization unless both raw clocks are explicitly supplied.

### SE(3) / geometry

For camera, hands, and objects:

- shape = 4×4
- finite values
- homogeneous bottom row
- rotation orthonormality error
- rotation determinant near +1
- translation speed and frame-to-frame jump
- rotation jump angle

### Task-state labels

- grasp ratio per hand
- grasp transitions
- finished-label ratio
- anchor changes
- dynamic object ratio

### Distribution visualization

The offline dashboard creates:

- image-quality histograms
- episode duration/frame-count comparison
- hand/camera/object XYZ trajectory plots
- speed distributions
- mask-coverage distributions
- a lightweight 2-D image-feature PCA map

The PCA map uses cheap image statistics/downsampled appearance features. It is for quick distribution screening, **not a semantic CLIP/DINO embedding replacement**.

For semantic embeddings, optionally launch FiftyOne:

```bash
pip install fiftyone
python humanego-data-quality/scripts/launch_fiftyone.py \
  --frame-csv quality_reports/<task>/latest/frame_metrics.csv \
  --dataset-name humanego_<task>
```

Then use FiftyOne/Brain separately for CLIP/DINO embeddings, UMAP, similarity, and duplicate analysis if desired.

## Quality score

`quality_score` is a triage score from 0–100, not a scientific benchmark. It combines penalties for:

- missing/corrupt data
- invalid transforms
- timestamp problems
- image quality problems
- duplicate/redundant frames
- trajectory jumps
- mask failures
- inconsistent task-state labels

The report also computes a lightweight train-vs-eval appearance mean-shift diagnostic when both splits are present. Treat it as a screening heuristic, not a semantic distance benchmark.

Never compare scores across projects unless they use the same config thresholds.

Suggested interpretation:

- `>= 90`: clean enough for normal training; inspect warnings.
- `80–89`: usable, but fix/review concentrated issues.
- `65–79`: likely to affect learning; clean or re-preprocess first.
- `< 65`: do not start a serious training run before diagnosis.

## HumanEgo-specific training split check

HumanEgo's common `data_sources` workflow holds recording `000` out for evaluation and trains on later recordings. If `--eval-session-pattern '*_000_vrs'` is enabled, compare eval vs train image/trajectory distributions and warn when the held-out episode is obviously out-of-distribution.

Do not "fix" a difficult eval episode by moving it into training unless the experimental protocol explicitly permits that. The correct fix may be broader demonstrations.

## Reprojection checks

Reprojection is deliberately adapter-based because PICO/Aria hand points may be stored in different coordinate frames.

To add it:

1. Copy `scripts/reprojection_adapter_example.py`.
2. Implement `load_world_points(frame_dir, training_json)` so it returns Nx3 points in the **same world frame as `metadata.c2w`**.
3. Run with:

```bash
python humanego-data-quality/scripts/audit_dataset.py ... \
  --reprojection-adapter /absolute/path/to/my_pico_reprojection_adapter.py
```

The audit projects world points with `w2c = inv(c2w)` and `metadata.k`, then reports valid projected points. If your camera coordinate convention differs from OpenCV `(x right, y down, z forward)`, convert inside the adapter. Never hide this conversion in the core audit.

## Agent decision rules

- If >5% frames have invalid/missing training JSON: stop and recommend re-preprocessing.
- If any episode has non-monotonic timestamps: inspect synchronization before training.
- If pose jumps are concentrated in one episode: quarantine/review that episode rather than globally smoothing every trajectory.
- If nearly all episodes have similar pose jumps: suspect coordinate/calibration/config issues.
- If image quality is fine but train/eval PCA clusters separate strongly: collect more diverse demonstrations before increasing model capacity.
- If train loss is low and eval loss is high, correlate eval failures with this report before declaring pure model overfitting.
- If duplicates are mainly adjacent video frames, consider temporal subsampling/stride; do not indiscriminately deduplicate action trajectories.

## Installation

Minimal audit dependencies:

```bash
pip install -r humanego-data-quality/requirements-quality.txt
```

The audit is CPU-capable. FiftyOne is optional and intentionally kept out of the minimal requirements.

## Expected final answer from the agent

A good response after running the skill should look like:

```text
Dataset: open_door
Episodes: 24 | Frames: 31,842
Quality score: 78.6/100 (heuristic)

Main issues:
1. mps_open_door_017_vrs: right-hand pose jumps at frames 421–438.
2. 11.8% sampled image frames are near-consecutive duplicates, mostly before contact.
3. Eval episode 000 is visually shifted from train episodes in the lightweight PCA map.
4. Object mask is nearly empty in 6.2% of episode 009.

Recommendation:
- Review/re-preprocess 017 and 009 first.
- Keep episode 000 as eval; collect 3–5 demonstrations matching its viewpoint/lighting.
- Re-run the audit, then train and compare eval loss again.
```
