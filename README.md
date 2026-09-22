# pico_humanego

这是一个基于 [HumanEgo](https://github.com/TX-Leo/HumanEgo) 的个人扩展版本。

本仓库不再重复介绍官方 HumanEgo 框架、论文和公开数据集，重点记录我在
HumanEgo 基础上完成的 PICO 数据接入、预处理、质量检查、仿真推理和实验配置。

## 我修改和新增的内容

### 1. PICO 数据导入与 HumanEgo 格式转换

新增 `tools/pico_import/`，将 PICO 录制数据转换为 HumanEgo 可以继续处理的
逐帧目录格式：

```text
humanego_ready/
└── preprocess/
    └── all_data/<frame>/
        ├── rgb.png
        ├── aria_cam_rgb.json
        ├── aria_hands.json
        ├── aria_slam.json
        └── aria_phases.json
```

主要功能包括：

- MP4、视频 PTS 和 `trackingData` 的时间同步；
- PICO 头显、相机和手部追踪数据的解析；
- 标定参数和坐标变换的显式记录；
- 双目 MediaPipe 2D 手部检测及可视化；
- 双目三角化、重投影和极线几何诊断；
- PICO 阶段划分，并复用 HumanEgo 的下游 `DatasetGen` 流程。

详细说明见 [`docs/pico_usage.md`](docs/pico_usage.md)。

### 2. HumanEgo 预处理流程适配

对部分原有预处理模块进行了适配和修正，包括：

- 阶段划分和运动状态处理；
- 相机三角化和坐标变换；
- DINO-SAM、LaMa、MediaPipe 手部处理；
- `DatasetGen` 和完整预处理流程；
- PICO 专用预处理配置，位于 `cfg/preprocess/base/`。

这些修改尽量保持原有 Aria 数据流程不变，并通过 PICO 导入层提供
HumanEgo 兼容的输入。

### 3. MuJoCo 推理和验证

新增 MuJoCo 推理适配代码：

- `inference/mujoco_backend.py`：仿真、渲染和相机坐标转换；
- `inference/mujoco_adapters.py`：相机、感知和机械臂接口；
- `inference/run_mujoco_inference.py`：开环预测和闭环控制验证；
- `cfg/inference/mujoco_serve_bread.yaml`：对应的场景和控制配置；
- `inference/task_grasp.py`、`inference/task_placement.py`：任务级推理入口。

示例：

```bash
conda activate humanego
MUJOCO_GL=egl python inference/run_mujoco_inference.py --dry-run --cycles 1
MUJOCO_GL=egl python inference/run_mujoco_inference.py --cycles 250
```

### 4. 数据质量检查工具

新增 `humanego-data-quality/`，用于只读检查 HumanEgo 或转换后的 PICO 数据，
包括：

- 图像质量、模糊度、亮度和熵；
- mask 覆盖率和手/物体轨迹；
- 时间戳、帧数和 episode 统计；
- 位姿、重复帧和异常帧检查；
- 训练集与评估集分布检查；
- HTML、CSV、JSON 和图表报告。

安装和使用：

```bash
pip install -r humanego-data-quality/requirements-quality.txt
python humanego-data-quality/scripts/audit_dataset.py \
  --session /path/to/humanego_ready/recording \
  --output ./quality_reports/local_check
```

该工具不会修改原始数据。完整参数见
[`humanego-data-quality/README.md`](humanego-data-quality/README.md)。

### 5. 训练配置和 AutoResearch 记录

新增或整理了以下内容：

- `open_door` 单手和左手训练配置；
- PICO 相关训练和预处理配置；
- MuJoCo `serve_bread` 推理配置；
- `autoresearch/` 下的搜索空间、实验配置、数据划分和结果记录；
- 训练器对显式数据路径和实验输出的支持。

实验记录只保留轻量级配置和文本结果，不上传训练数据或 checkpoint。

### 6. 测试、工具和文档

新增内容包括：

- `tests/test_pico_import.py`；
- `tests/test_pico_stereo_geometry.py`；
- PICO 视频、追踪和几何工具；
- PICO 迁移审计、里程碑报告和使用文档；
- MediaPipe 视频可视化脚本。

## 推荐工作流

```text
PICO 原始录制
    │
    ├── MP4 + trackingData + PTS
    │
    ▼
tools/pico_import/recording.py
    │
    ▼
humanego_ready/preprocess/all_data/
    │
    ├── PICO 阶段和双目几何验收
    ├── HumanEgo 预处理 / DatasetGen
    ├── 数据质量检查
    └── 训练或 MuJoCo 推理验证
```

PICO 导入的基本命令：

```bash
python -m tools.pico_import.recording \
  /path/to/CameraRecord.mp4 \
  /path/to/trackingData.txt \
  /path/to/CameraRecord.jsonl \
  --output ./pico_import_stage/recording_id
```

导出 HumanEgo 兼容数据：

```bash
python -m tools.pico_import.humanego_writer \
  /path/to/CameraRecord.mp4 \
  /path/to/trackingData.txt \
  /path/to/CameraRecord.jsonl \
  --output ./humanego_ready/recording_id
```

继续运行下游预处理：

```bash
python -m tools.pico_import.run_downstream \
  --input ./humanego_ready/recording_id \
  --reference-frame 350
```

PICO 双目验收和坐标系说明见：

- [`docs/pico_usage.md`](docs/pico_usage.md)
- [`docs/pico_port_audit.md`](docs/pico_port_audit.md)
- [`docs/pico_v3_milestone1_report.md`](docs/pico_v3_milestone1_report.md)
- [`docs/pico_v3_milestone2_geometry_report.md`](docs/pico_v3_milestone2_geometry_report.md)

## 目录说明

| 目录 | 内容 |
|---|---|
| `tools/pico_import/` | PICO 导入、同步、标定、三角化和 HumanEgo 导出 |
| `humanego-data-quality/` | 数据质量检查和离线报告 |
| `preprocess/` | HumanEgo 预处理及 PICO 适配修改 |
| `inference/` | 真实机器人和 MuJoCo 推理代码 |
| `cfg/` | 训练、预处理和推理配置 |
| `autoresearch/` | 配置搜索和实验记录 |
| `docs/` | PICO 迁移文档和验收报告 |
| `tests/` | PICO 导入与双目几何测试 |

## 数据和文件管理

数据集、原始录制、验证帧、视频、训练输出、checkpoint、模型权重和质量报告
不进入 Git。它们由根目录的 [`.gitignore`](.gitignore) 排除，运行时请通过本地
路径提供。

当前仓库只保存代码、配置、文档、测试和必要的轻量资源。

## 当前限制

- PICO 导入需要同时提供视频、追踪数据和时间戳信息；缺少其中任一项时不会
  静默生成伪造数据。
- 相机内参和头显到相机的外参必须来自经过验证的标定结果。
- PICO 双目 2D 检测和 3D 诊断通过后，才能继续使用下游 DatasetGen。
- 真实机器人运行仍需要用户自行提供硬件驱动、手眼标定和安全配置。
- 测试和完整预处理需要安装项目依赖；质量检查工具的依赖见
  `humanego-data-quality/requirements-quality.txt`。

## 说明

本仓库保留上游 HumanEgo 的 `LICENSE`。HumanEgo 原始论文、官方数据集和
官方使用方式请以其上游仓库为准；本 README 只记录本项目的修改部分。
