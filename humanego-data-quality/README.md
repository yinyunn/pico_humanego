# humanego-data-quality

A read-only HumanEgo/PICO→HumanEgo dataset audit skill with an offline visual report.

## Install

```bash
cd HumanEgo
pip install -r humanego-data-quality/requirements-quality.txt
```

## Run a task

```bash
python humanego-data-quality/scripts/audit_dataset.py \
  --data-root ./data \
  --task serve_bread \
  --output quality_reports/serve_bread/latest
```

## Run one recording

```bash
python humanego-data-quality/scripts/audit_dataset.py \
  --session ./data/serve_bread/aria/mps_serve_bread_001_vrs \
  --output quality_reports/serve_bread/001
```

## Large dataset smoke test

```bash
python humanego-data-quality/scripts/audit_dataset.py \
  --data-root ./data --task open_door \
  --image-stride 10 \
  --output quality_reports/open_door/smoke
```

Open `report.html` in a browser. Detailed artifacts are `frame_metrics.csv`, `episode_metrics.csv`, `issues.csv`, `summary.json`, and `plots/`.

## Optional FiftyOne view

```bash
pip install fiftyone
python humanego-data-quality/scripts/launch_fiftyone.py \
  --frame-csv quality_reports/open_door/latest/frame_metrics.csv \
  --dataset-name humanego_open_door
```

The offline report works without FiftyOne.
