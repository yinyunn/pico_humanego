<h3 align="center">🎉 HumanEgo is now fully released — code, dataset &amp; docs are all live! (June 7, 2026) 🎉</h3>

<p align="center">
  <a href="https://humanego-ai.github.io">
    <img src="assets/title/hero.png" alt="HumanEgo — Zero-Shot Robot Learning from Minutes of Human Egocentric Videos" width="100%" />
  </a>
</p>

<p align="center">
  <a href="https://tx-leo.github.io">Zhi (Leo) Wang</a> &nbsp;·&nbsp;
  <a href="https://bottle101.github.io/">Botao He</a> &nbsp;·&nbsp;
  <a href="https://colinyu1.github.io/">Kelin Yu</a> &nbsp;·&nbsp;
  <a href="https://sjlee.cc/">Seungjae Lee</a> &nbsp;·&nbsp;
  <a href="https://ruohangao.github.io/">Ruohan Gao</a> &nbsp;·&nbsp;
  <a href="https://furong-huang.com/">Furong Huang</a> &nbsp;·&nbsp;
  <a href="https://robotics.umd.edu/clark/faculty/350/Yiannis-Aloimonos">Yiannis Aloimonos</a>
</p>

<p align="center">
  <a href="https://humanego-ai.github.io"><img src="assets/title/btn_website.png" alt="Website" height="60" /></a>
  &nbsp;
  <a href="https://arxiv.org/pdf/2605.24934"><img src="assets/title/btn_paper.png" alt="Paper" height="60" /></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2605.24934"><img src="assets/title/btn_arxiv.png" alt="arXiv" height="60" /></a>
  &nbsp;
  <a href="https://youtu.be/pdL46diijuY"><img src="assets/title/btn_video.png" alt="Video" height="60" /></a>
  &nbsp;
  <a href="#bibtex"><img src="assets/title/btn_bibtex.png" alt="BibTeX" height="60" /></a>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/Leo-TX/HumanEgo"><img src="https://img.shields.io/badge/Dataset-Leo--TX%2FHumanEgo-ff8c2b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset" /></a>
  &nbsp;
  <a href="https://huggingface.co/Leo-TX/HumanEgo"><img src="https://img.shields.io/badge/Checkpoints-Leo--TX%2FHumanEgo-e8772e?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Checkpoints" /></a>
  &nbsp;
  <a href="https://leo-tx-humanego-gallery.static.hf.space"><img src="https://img.shields.io/badge/Data_Gallery-Browse_122_clips-ff6a00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Data Gallery" /></a>
</p>

<p align="center">
  <img src="https://komarev.com/ghpvc/?username=tx-leo-humanego&base=5000&label=Views&color=E8772E&style=for-the-badge" alt="Repo views" />
</p>

---

## Overview

<p align="center">
  <img src="assets/teaser.gif" alt="HumanEgo teaser" width="100%" />
</p>

There are three ways to use this repo, in increasing order of effort:

1. **[Quick Start in 5 Minutes](#quick-start-in-5-minutes)** — run the whole pipeline
   end-to-end on two sample recordings, as a smoke test.
2. **[Train on the HumanEgo Dataset](#train-on-the-humanego-dataset)** — download
   our full released data (with precomputed labels) and train, no hardware needed.
3. **[Train Your Own Policy](#train-your-own-policy)** — collect your own
   egocentric demonstrations with Project Aria glasses and train on them.

---

## Installation

```bash
git clone https://github.com/TX-Leo/HumanEgo.git
cd HumanEgo
conda create -n humanego python=3.11 -y
conda activate humanego
bash setup.sh
```

By default this installs everything the released pipeline needs: PyTorch (with
CUDA) and the vision foundation models we use (SAM 2, Grounding DINO, CoTracker,
Orient-Anything V2). The pipeline relies on Project Aria's built-in MPS hand
tracking, so the alternative hand-detection methods (MediaPipe, WiLoR, HaMeR) and
the robot/camera hardware drivers are **skipped by default** to keep the install
lean. Enable them per-run only if you need them:

```bash
SKIP_HAND=0     bash setup.sh   # + MediaPipe / WiLoR / HaMeR (alternative hand-tracking methods)
SKIP_HARDWARE=0 bash setup.sh   # + pyrealsense2 / trossen-arm (real-robot collection & deployment)
PREDOWNLOAD=1   bash setup.sh   # pre-download model weights now (else fetched on first run)
```

---

## Quick Start in 5 Minutes

<table align="center"><tr>
<td align="center" width="50%"><img src="assets/aria_vis_serve_bread.gif" width="100%" /><br><sub><code>serve_bread</code> — preprocessing visualization</sub></td>
<td align="center" width="50%"><img src="assets/aria_vis_water_flowers.gif" width="100%" /><br><sub><code>water_flowers</code> — preprocessing visualization</sub></td>
</tr></table>

The fastest way to run the whole pipeline end-to-end — download, preprocess, and
train on just a couple of recordings. The `HumanEgo` training job holds out the
first recording (`mps_serve_bread_000_vrs`) for evaluation and trains on the
rest, so download two.

**1. Download two recordings** — inputs only, ~1.2 GB

```bash
python scripts/download_data.py --task serve_bread --num 2 --input-only
```

Fetches `mps_serve_bread_000_vrs` and `mps_serve_bread_001_vrs` into
`./data/serve_bread/aria/`, skipping the precomputed `preprocess/` output so you
run the pipeline yourself. See
[Train on the HumanEgo Dataset](#train-on-the-humanego-dataset) for the full
dataset and all download options.

**Prefer to skip preprocessing?** Drop `--input-only` to download the two recordings
**with the precomputed `preprocess/` output** (~4 GB, auto-extracted), then skip
Step 2 and jump straight to **3. Train**:

```bash
python scripts/download_data.py --task serve_bread --num 2
```

**2. Preprocess both**

```bash
python -m preprocess.Preprocess --mps_path ./data/serve_bread/aria/mps_serve_bread_000_vrs --task serve_bread
python -m preprocess.Preprocess --mps_path ./data/serve_bread/aria/mps_serve_bread_001_vrs --task serve_bread
```

Regenerates each recording's `preprocess/` folder. See
[Step 2: Preprocessing](#step-2-preprocessing) for details.

**3. Train**

```bash
python -m training.FlowMatchingTrainer --task serve_bread --use_cfg --job HumanEgo
```

Trains on `mps_serve_bread_001_vrs` and evaluates on the held-out
`mps_serve_bread_000_vrs` (config: `cfg/training/serve_bread/HumanEgo.yaml`).
See [Step 3: Training](#step-3-training) for details.

---

## Train on the HumanEgo Dataset

<table align="center"><tr>
<td align="center" width="50%"><img src="assets/infer_serve_bread.gif" width="100%" /><br><sub><code>serve_bread</code> — learned policy on a real robot</sub></td>
<td align="center" width="50%"><img src="assets/infer_water_flowers.gif" width="100%" /><br><sub><code>water_flowers</code> — learned policy on a real robot</sub></td>
</tr></table>

Skip data collection entirely: download our full released dataset — raw Aria
recordings **and** the precomputed MPS + preprocess output — and train directly.
Everything is hosted on the public HuggingFace dataset
[`Leo-TX/HumanEgo`](https://huggingface.co/datasets/Leo-TX/HumanEgo), no login or
token required. We release two tasks: **`serve_bread`** and **`water_flowers`**.

### Download the full dataset

```bash
# everything, both tasks, with precomputed preprocess output (large)
python scripts/download_data.py --task all --num all

# or one task at a time
python scripts/download_data.py --task serve_bread   --num all
python scripts/download_data.py --task water_flowers --num all
```

Each recording lands at `./data/<task>/aria/mps_<task>_<id>_vrs/` with its
`preprocess/` folder already populated (the `all_data.tar` is auto-extracted).
Use `--num N` for the first N recordings, or `--input-only` to skip the
precomputed output and run preprocessing yourself. See
[`preprocess/README.md`](preprocess/README.md) for the full output-file
reference and a plain-`huggingface_hub` recipe.

### Train

```bash
# serve_bread
python -m training.FlowMatchingTrainer --task serve_bread   --use_cfg --job HumanEgo

# water_flowers
python -m training.FlowMatchingTrainer --task water_flowers --use_cfg --job HumanEgo
```

Each job holds out recording `000` of the task for evaluation and trains on the
rest, reading `cfg/training/<task>/HumanEgo.yaml`. See
[Step 3: Training](#step-3-training) for the `--task` / `--job` convention.

---

## Train Your Own Policy

Collect your own human-egocentric demonstrations and train a policy on them,
end-to-end — record with Project Aria glasses, process the data through MPS,
preprocess it, train, and deploy.

### Step 1: Data Collection

<p align="center">
  <img src="assets/data_collection.gif" alt="HumanEgo data collection — anyone, anytime, anywhere, with only 30 minutes of data" width="100%" />
</p>

To apply for the Meta Project Aria glasses, see
[projectaria.com/glasses](https://www.projectaria.com/glasses/).

We are also actively adding support for other hardware — e.g. Meta Aria Gen 2,
Meta Quest, Apple Vision Pro, RealSense, iPhone, etc. For the Quest setup in
particular, see
[`LV-Robotics-Lab/quest_streamer`](https://github.com/LV-Robotics-Lab/quest_streamer).

See [`datacollection/README.md`](datacollection/README.md)
for the end-to-end guide on recording your own Project Aria data and running
MPS (SLAM + hand tracking) on it. The resulting data should look like this:

```
- data
    - mps_TEST_vrs/
        - else
            - sample.vrs.json
            - vrs_health_check.json
            - vrs_health_check_slam.json
        - hand_tracking
            - hand_tracking_results.csv
            - summary.json
        - slam
            - closed_loop_trajectory.csv
            - online_calibration.jsonl
            - open_loop_trajectory.csv
            - semidense_observations.csv.gz
            - semidense_points.csv.gz
            - summary.json
        - sample.vrs
```

### Step 2: Preprocessing

<p align="center">
  <img src="assets/data_collection.webp" alt="HumanEgo preprocessing visualization" width="100%" />
</p>

Turn raw MPS output into training-ready data. First, create a task config
`cfg/preprocess/tasks/<your_task>.yaml` describing **your** task — the
open-vocabulary detection prompts for each object, which hand(s) to track, etc.
`--task <your_task>` merges it over the defaults in `cfg/preprocess/base/`. See
[Adding a new task](preprocess/README.md#52-adding-a-new-task) for the
field-by-field reference. Then point `--mps_path` at the MPS folder from Step 1 and run:

```bash
python -m preprocess.Preprocess --mps_path ./data/<your_mps_folder> --task <your_task>
```

This regenerates everything under `…/preprocess/`. See
**[`preprocess/README.md`](preprocess/README.md)** for the full data layout,
the output-file reference, the task-config reference, and download options.

### Step 3: Training

<p align="center">
  <img src="assets/architecture.webp" alt="HumanEgo training architecture" width="100%" />
</p>

Train a flow-matching policy on the preprocessed data:

```bash
python -m training.FlowMatchingTrainer --task "YOUR_TASK" --use_cfg --job "YOUR_JOB"
```

`--task` selects the data + config folder under `cfg/training/` and `--job` selects a
YAML inside it (e.g. `HumanEgo` → `cfg/training/serve_bread/HumanEgo.yaml`); outputs go
to `runs/<task>/<job>/`.

**To train on your own task:** preprocess your recordings (Step 2 — you need ≥2, one is
held out for evaluation), then create `cfg/training/<your_task>/HumanEgo.yaml` (copy
`cfg/training/serve_bread/HumanEgo.yaml` and set `data_sources`, `single_hand`, etc.).
See **[`training/README.md`](training/README.md)** for what data training expects, the
full parameter reference, and how to add your own config.

### Step 4: Inference

Deploy a trained policy on a real dual-arm robot. Every control step the policy
consumes a **clean, embodiment-agnostic image** (the real arm inpainted out, a
virtual gripper rendered in its place) and **Interaction-Centric Tokens (ICT)**
(every hand and object as a 6DoF entity), and predicts a future end-effector
trajectory that is smoothed and servoed to the arms in a closed loop:

```
camera ─▶ perception ─▶ clean image + ICT ─▶ policy ─▶ EE trajectory ─▶ robot ─▶ (loop)
```

```bash
# install the hardware drivers (RealSense + Trossen) first:
SKIP_HARDWARE=0 bash setup.sh
# then run the dual-arm reference loop:
python inference/run_inference.py cfg/inference/example_dualarm.yaml
```

**Pretrained checkpoints (no training required).** Our released policies live in the
public HuggingFace **model** repo
[`Leo-TX/HumanEgo`](https://huggingface.co/Leo-TX/HumanEgo). Each task folder ships
`latest.pt` (inference weights) next to the `config.json` + `dataset_stats.json` the
loader reads from the same directory. Download the `serve_bread` policy with:

```bash
huggingface-cli download Leo-TX/HumanEgo --include "serve_bread/*" --local-dir ./checkpoints
# → ./checkpoints/serve_bread/{latest.pt, config.json, dataset_stats.json}
```

Then set `policy.ckpt: ./checkpoints/serve_bread/latest.pt` in your inference config
(`cfg/inference/example_dualarm.yaml`). The same folder also carries `latest_full.pt`
(optimizer state) if you'd rather resume training.

The [`inference/`](inference/README.md) folder is a clean, hardware-agnostic
**reference template** — it shows the standard structure rather than a turn-key
script. Implement three interfaces (`Camera`, `RobotArm`, `Perception`) for your
own rig and reuse the policy + control logic unchanged. You'll need a trained
checkpoint, a camera + arm(s), and a hand-eye calibration (`T_base_in_cam`). See
**[`inference/README.md`](inference/README.md)** for the full walk-through: frame
conventions, the step-by-step pipeline, how to write your own
camera/robot/perception drivers, and the parameter-tuning guide.

---

## Acknowledgements

This project builds on excellent open-source work, including
[Project Aria](https://www.projectaria.com/) (Gen 1 glasses &amp;
[MPS](https://facebookresearch.github.io/projectaria_tools/docs/intro)),
[Trossen Arm](https://www.trossenrobotics.com/),
[CoTracker3](https://github.com/facebookresearch/co-tracker),
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
[SAM 2](https://github.com/facebookresearch/sam2),
[HaMeR](https://github.com/geopavlakos/hamer),
[WiLoR](https://github.com/rolpotamias/WiLoR),
[MediaPipe](https://github.com/google-ai-edge/mediapipe),
[LaMa](https://github.com/advimman/lama),
and [Orient-Anything](https://github.com/SpatialVision/Orient-Anything).

---

## License

HumanEgo is released under the [PolyForm Noncommercial License 1.0.0](LICENSE): free
for any **noncommercial** use, including academic and nonprofit research. **Commercial
use (by or for a company) requires a separate paid license** — please get in touch
(see [Contact](#contact)).

---

## Contact

<p align="center">
  <img src="assets/team.png" alt="The HumanEgo team — University of Maryland" width="100%" />
</p>

Questions are welcome! Reach out to Zhi (Leo) Wang at
[tx.leo.wz@gmail.com](mailto:tx.leo.wz@gmail.com) (WeChat: `tx-leo-wz`).

**Join our WeChat group** to discuss HumanEgo, ask questions, and stay updated — scan the QR code below:

<p align="center">
  <img src="assets/WeChat.jpg" alt="HumanEgo WeChat group QR code" width="260" />
</p>

> If the QR code has expired, add Zhi (Leo) Wang on WeChat (`tx-leo-wz`) and we'll invite you to the group.

---

<h2 id="bibtex">BibTeX</h2>

If you find this work helpful, we would greatly appreciate it if you cite our paper!

```bibtex
@misc{humanego2026,
  title         = {HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos},
  author        = {Wang, Zhi and He, Botao and Yu, Kelin and Lee, Seungjae and Gao, Ruohan and Huang, Furong and Aloimonos, Yiannis},
  year          = {2026},
  eprint        = {2605.24934},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```
