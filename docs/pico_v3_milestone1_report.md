# PICO v3 Milestone 0–1 验收报告

日期：2026-09-09。完成现有流程审计、native baseline 保存、同帧双目 MediaPipe 2D smoke。
按任务说明第 41 节停在 2D 人工验收点；**尚未完成 stereo 3D、final hand source 替换和 interaction phase**。

## 修改范围

真实结构和 16 项数据流问题详见 `pico_existing_pipeline_before_v3.md`。

| 已有文件 | 改动 |
|---|---|
| `preprocess/MediaPipeHands.py` | 将现有模型初始化提取为 `create_hand_landmarker`；Aria 默认 IMAGE 模式及阈值不变，PICO 使用 VIDEO 模式和本地同一模型 |
| `scripts/visualize_mediapipe_mp4.py` | 复用 crop 和模型 helper，读实际 PTS，保存同步信息、浮点 2D、QA 和预览；候选配对标记未验证/歧义，移除 z_norm |
| `tools/pico_import/humanego_writer.py` | 新增 `--stereo-2d-only` 分支；不写 aria_hands、phase 或 training_data |
| `tests/test_pico_import.py` | 增加 RGB/像素坐标/无深度输出、匹配歧义、smoke 不写最终标签的检查 |
| `docs/pico_usage.md` | 添加运行与输出说明 |

新增文件仅两份审计/验收文档，无新 adapter 或 backend。生成数据和源码快照放在 outputs 下。

旧主路径：TXT 26 joints → `_pack_hand` → native gripper/grasp → head-motion phase → downstream。
本轮新分支：同帧 SBS MP4 → 既有 crop → 两个独立 VIDEO detector → 2D+PTS+QA。
主路径尚未切换，避免把未验证的手性对应关系当作 metric geometry。

## 真实运行命令

在 HumanEgo 根目录、humanego 环境执行：

```bash
python -m tools.pico_import.humanego_writer \
  /home/yyn/workspace/pico_download/test/Download/CameraRecord_20260908_174122.mp4 \
  /home/yyn/workspace/pico_download/test/Download/trackingData_20260908_174122.txt \
  /home/yyn/workspace/pico_download/test/humanego_data/_pico_pts/pts/CameraRecord_20260908_174122.jsonl \
  --output outputs/pico_v3_smoke/recording_20260908_174122 \
  --stereo-2d-only
```

输出已存在，复跑请选新的 output。
基线命令使用相同三个输入，去掉 `--stereo-2d-only`，output 为
`outputs/pico_v3_baseline/recording_20260908_174122`。
确切 argv 在 `outputs/pico_v3_baseline/baseline_command.json`。

## Artifact 与 QA

所有路径相对 HumanEgo：

- `outputs/pico_v3_baseline/git_status_before.txt`、`git_head.txt`、`source_before/`：修改前状态与源码。
- `outputs/pico_v3_baseline/baseline_run.log`：原 writer 重跑成功，765 帧中保留 763 帧，2 帧同步超限。
- `outputs/pico_v3_baseline/recording_20260908_174122/preprocess/vis/native_hand_gripper.mp4`
  和 `native_hand_gripper_contact_sheet.jpg`：native skeleton + 当前 gripper frame 轴 + phase 文本。
- 同 baseline 的 `preprocess/aria_phases_analysis.png`：原 phase 曲线。
- `outputs/pico_v3_baseline/existing_open_door_20260827_212621/`：归档已跑通 Open Door 的 JSON，
  包含原 training_data。**它是既有输出归档，不是本轮重新运行完整下游。**
- `outputs/pico_v3_smoke/recording_20260908_174122/stereo_2d/mediapipe_stereo_vis.mp4`：完整 2D 视频。
- 同目录 `mediapipe_stereo_2d.contact_sheet.jpg`、`mediapipe_stereo_2d.jsonl`、
  `mediapipe_stereo_2d.summary.json`、`qa_review.json`：可视化、原始 2D 和统计。
- `outputs/pico_v3_baseline/stereo_2d_run.log`：完整实测日志。

| 指标 | 值 |
|---|---:|
| 双目视频 | 2160 × 810，765 帧 |
| 每目尺寸 | 1080 × 810 |
| 左目手检测次数 | 1371 |
| 右目手检测次数 | 1376 |
| 按手性生成候选对 | 1179（未几何验证） |
| 左目完全无手检测帧比例 | 3.40% |
| 右目完全无手检测帧比例 | 2.88% |
| 为 VIDEO API 修正重复毫秒时间戳次数 | 0 |

无手检测比例不是检测错误率：手在画外也计入，不能据此声称准确率 96% 以上。
检查了六个时间点的左右预览；张手、握拳可见较完整 landmark。
**第 611 帧右目将两只手都标成 Right**，证明 handedness alone 不足以确定对应关系。
后续必须增加 epipolar/wrist consistency 并拒绝歧义。

## 验证与尚未确认事项

- 8 项测试函数通过：既有 calibration/crop/tracking/PTS 检查及新增 3 项。
  当前环境没有 pytest，使用直接函数 runner、临时目录和 unittest.mock 执行，未安装新依赖。
- 765 帧 JSON 与源 PTS 逐项一致；输出 MP4 帧数一致；每次检测 21 点且没有 MP z；
  全部匹配标记 geometry_verified=false。
- 共享模型 IMAGE 模式真实推理成功，样本检出两只手；VIDEO 模式完整录制成功。
  原 Aria 整段 preprocessing 未重跑，不能视为完整 Aria 回归通过。
- 现有 camera/head/parser、native `_pack_hand`、phase、下游和训练逻辑保持原状。
  原始数据及已有录制输出没有覆盖，baseline 与 smoke 使用独立目录。
- 当前视频是否在录制端去畸变/校正仍需确认。`d=[]` 不是证据。
- 本轮没有 metric 3D、重投影误差、静态 jitter、phase F1 或 policy success 指标。
  Open Door/Bread→Plate 的新端到端验收属于后续 milestones，不能宣称已完成。

下一步需先检查本轮左右 2D 预览，再按第 42 节接同帧三角化及 gripper；
hand 验证后才按第 43 节替换现有 phase。
