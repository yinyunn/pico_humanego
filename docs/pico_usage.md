# PICO 原始数据导入

当前适配分为两个阶段：同步中间层，以及导出 HumanEgo 兼容的
`preprocess/all_data`。导出阶段会生成相机、手部、Head 轨迹和 Pico 阶段文件。

## v3：双目 MediaPipe 2D 验收

目前已增加第一阶段 smoke 入口，复用现有 crop、同步和 MediaPipe 模型加载。
这一入口仅生成 2D 检测与可视化；默认导出的 native hand 和 phase 尚未替换。

```bash
cd /home/yyn/workspace/HumanEgo
conda run --no-capture-output -n humanego python -m tools.pico_import.humanego_writer \
  /path/to/CameraRecord.mp4 \
  /path/to/trackingData.txt \
  /path/to/CameraRecord.jsonl \
  --output outputs/pico_v3_smoke/my_recording \
  --stereo-2d-only
```

可加 `--max-frames 120` 限制帧数。使用现有
`weights/mediapipe/hand_landmarker.task`，不自动下载另一份模型。
输出目录下的 `stereo_2d` 必须尚不存在，以免覆盖前一次验收结果。

- `mediapipe_stereo_vis.mp4`：同一解码帧的左右目检测。
- `mediapipe_stereo_2d.jsonl`：左右目局部/全幅浮点像素、手性及分数、PTS、tracking 同步信息。
- `mediapipe_stereo_2d.contact_sheet.jpg`：六个时间点预览。
- `mediapipe_stereo_2d.summary.json`：检测数、无手检测帧占比、时间戳来源。
- `calibration.json`、`sync_manifest.jsonl`、`sync_summary.json`：既有同步中间层输出。

`stereo_matches` 只表示手性候选，`geometry_verified=false`；`ambiguous=true`
表示同一目有重复手性。它们不能直接输入三角化。`score` 是手性分类分数，
不是关节可见性或 3D 置信度。同步超限帧保留供可视化，但有明确标记。
JSON 不输出 MP z 或 world landmarks，也不写最终 hand label。

审计及后续计划见 [pico_existing_pipeline_before_v3.md](pico_existing_pipeline_before_v3.md)，
本轮实测结果见 [pico_v3_milestone1_report.md](pico_v3_milestone1_report.md)。

## v3：双目几何诊断

确认 2D 后可运行同帧三角化诊断。该命令严格采用：

```text
E: raw camera -> head
D: raw camera -> image camera
p_head = E @ inv(D) @ p_image
```

`D` 只改变坐标基，不表示相机安装位置。诊断不会拟合、反转或覆盖外参，
也不会写入 `aria_hands.json`。

```bash
python -m tools.pico_import.humanego_writer \
  /path/to/CameraRecord.mp4 /path/to/trackingData.txt /path/to/CameraRecord.jsonl \
  --output outputs/pico_v3_geometry_audit/my_recording \
  --stereo-3d-diagnostic /path/to/mediapipe_stereo_2d.jsonl
```

输出包括 `epipolar_geometry_contact_sheet.jpg`、`epipolar_geometry.mp4`、
`stereo_reprojection_contact_sheet.jpg`、`stereo_reprojection.mp4`、
逐帧 `stereo_3d.jsonl` 和 `summary.json`。只有全部 21 点通过正深度、
极线、双目重投影、ray angle、骨长和手尺寸检查时，诊断 gripper 才会显示。
当前真实数据结论见 [pico_v3_milestone2_geometry_report.md](pico_v3_milestone2_geometry_report.md)。

## 目录结构

```text
tools/pico_import/
├── recording.py       # 导入编排与视频/tracking 时间同步
├── video_reader.py    # MP4 PTS 和左右目 crop
├── tracking_reader.py # trackingData JSONL 解析
├── calibration.py     # KA、E1、D 和标定快照
├── pico_phases.py     # 复用 AriaPhasesOps 的 Pico 阶段划分
└── run_downstream.py  # DINO-SAM2/CoTracker/三角化/DatasetGen 编排
```

## 执行同步

```bash
cd /home/yyn/workspace/HumanEgo
conda run -n humanego python -m tools.pico_import.recording \
  /path/to/CameraRecord_YYYYMMDD_HHMMSS.mp4 \
  /path/to/trackingData_YYYYMMDD_HHMMSS.txt \
  /path/to/CameraRecord_YYYYMMDD_HHMMSS.jsonl \
  --output ./pico_import_stage/recording_id
```

中间输出：

- `sync_manifest.jsonl`：每个视频帧对应的 tracking 行和同步误差；
- `sync_summary.json`：同步覆盖率、平均误差和异常帧状态；
- `calibration.json`：当前适配参数 `K=KA`、左目 `E1`、`D` 及公式。

当前坐标约定为：

```text
H = Head.pose（direct）
HandJointLocations = app/tracking world
camera = inv(H @ E1) @ hand_point
image_camera = D @ camera
```

同步结果通过检查后，可以继续导出 HumanEgo 基础输入。

## 导出 HumanEgo 基础输入

同步验收后，可以生成 HumanEgo 现有逐帧目录格式：

```bash
conda run -n humanego python -m tools.pico_import.humanego_writer \
  /path/to/CameraRecord_YYYYMMDD_HHMMSS.mp4 \
  /path/to/trackingData_YYYYMMDD_HHMMSS.txt \
  /path/to/CameraRecord_YYYYMMDD_HHMMSS.jsonl \
  --output ./humanego_ready/recording_id
```

每一帧会生成：

```text
preprocess/all_data/00000/
├── rgb.png
├── aria_cam_rgb.json
├── aria_hands.json
├── aria_slam.json
└── aria_phases.json
```

`humanego_writer` 在写完所有帧的 `aria_slam.json` 和 `aria_hands.json` 后，
立即运行 `pico_phases.py`。它复用 `cfg/preprocess/base/AriaPhases.yaml`
和 `preprocess/AriaPhasesOps.py`，生成：

```text
preprocess/aria_phases_results.json
preprocess/aria_phases_analysis.png
preprocess/all_data/<idx>/aria_phases.json
```

Pico 阶段模式编码与 HumanEgo 保持一致：`0=MANIPULATION`、`1/2=NAVIGATION`
（对应 Aria 的 FORWARD/ROTATE）、`3=TRANSITION`、`4=FINISHED`。
若需要单独重算阶段，可以执行：

```bash
conda run -n humanego python -m tools.pico_import.pico_phases \
  --input ./humanego_ready/recording_id
```

阶段文件生成后，再运行下游处理：

```bash
conda run -n humanego python -m tools.pico_import.run_downstream \
  --input ./humanego_ready/recording_id \
  --reference-frame 350
```

`run_downstream` 最后调用 `DatasetGen`。只有到这一步，才会在每个帧目录中生成
`training_data.json`；它读取该帧的 `aria_phases.json`，因此 `mode=4` 的帧会被
写成 `metadata.is_finished=1.0`。

同步超出阈值的帧默认丢弃并写入 `sync_manifest.jsonl`；如需保留，可增加
`--keep-sync-outliers`。导出结果由 `humanego_import_summary.json` 标记状态。
