# 推理——真实机器人部署

本目录提供一个参考模板，用于将训练好的 HumanEgo 策略部署到真实双臂机器人上。模板有意保持简洁并与具体硬件解耦，展示 HumanEgo 推理栈的标准结构。你只需接入自己的相机、机械臂和感知模块，其余部分即可复用。

> ⚠️ **该模板不能开箱即用。** 它依赖真实硬件（相机和机械臂）、手眼标定结果以及计算量较大的感知模型。请将它视为搭建部署系统的蓝图，而不是直接运行的成品脚本。示例采用论文中的硬件配置：**Intel RealSense + Trossen 机械臂**，感知部分使用 **DINO-SAM + LaMa**。

---

## 核心思路

```text
 相机 ─▶ 感知 ─▶ 干净图像 + ICT ─▶ 策略 ─▶ 末端轨迹 ─▶ 机器人
   ▲                                                        │
   └────────────────── 闭环反馈 ◀───────────────────────────┘
```

HumanEgo 策略在每个控制周期接收两类输入，并预测未来的末端执行器轨迹：

1. **与机器人本体无关的干净 RGB 图像**：通过图像修复移除真实机械臂，并在原位置渲染虚拟夹爪，从而缩小人类视频训练数据和机器人部署图像之间的视觉差异。
2. **以交互为中心的 Token（Interaction-Centric Tokens，ICT）**：将每只手和每个物体表示成一个 6DoF 实体，同时编码每只手相对于该实体的位姿，从而缩小不同人手和机器人之间的运动学差异。

训练和推理阶段使用完全一致的 ICT 与干净图像构造方式，因此只使用人类第一视角视频训练的策略可以迁移到机器人上。

---

## 文件说明

| 文件 | 作用 |
|---|---|
| [`interfaces.py`](interfaces.py) | 需要实现的三个抽象接口：`Camera`、`RobotArm` 和 `Perception`。主循环只依赖这些接口。 |
| [`policy.py`](policy.py) | `ICTPolicy`：加载 checkpoint、执行 `prepare_image`、构建单/双臂 ICT、进行流匹配推理，并将结果解码为相机坐标系下的末端目标。 |
| [`controller.py`](controller.py) | `TrajectoryController`：通过 EMA、Slerp 和步长限制平滑预测轨迹，以滚动时域方式控制机械臂。 |
| [`run_inference.py`](run_inference.py) | 连接整个推理栈的主循环，包含示例硬件适配器和参考感知实现。 |
| [`../cfg/inference/example_dualarm.yaml`](../cfg/inference/example_dualarm.yaml) | 带注释的完整双臂推理配置。 |
| `CamRS.py`、`RobotArmTrossen.py` | RealSense 和 Trossen 的示例驱动，可据此适配自己的硬件。 |
| [`mujoco_backend.py`](mujoco_backend.py) | MuJoCo 共享后端，负责渲染、物理步进和坐标变换。 |
| [`mujoco_adapters.py`](mujoco_adapters.py) | MuJoCo 版本的相机、感知和机械臂接口。 |
| [`run_mujoco_inference.py`](run_mujoco_inference.py) | `serve_bread` 权重的 MuJoCo 开环/闭环验证入口。 |

---

## 首先确认：坐标系与单位约定

大部分部署问题都来自坐标系错误。接口约定如下，详细定义见 `interfaces.py`：

- 所有位姿均为 **4×4 的 SE(3) 齐次矩阵**，位置单位为米，旋转矩阵必须为合法旋转。
- 名称后缀 **`_in_cam` 表示位姿位于相机光学坐标系**。该坐标系采用 OpenCV 约定：+x 向右、+y 向下、+z 向前，并作为单个 episode 内共享的“世界”坐标系。
- 每台机械臂都必须提供 **`T_base_in_cam`**，表示机械臂基座坐标系在相机坐标系中的位姿。它来自手眼标定；外参错误是“机器人朝错误位置运动”的首要原因。
- **`T_align`** 用于将机械臂自身的末端坐标系对齐到模型训练时使用的“手坐标系”。如果策略使用相同末端约定的机器人/遥操作数据训练，可使用单位矩阵；使用发布的 Aria-MPS checkpoint 时，应采用 [`example_dualarm.yaml`](../cfg/inference/example_dualarm.yaml) 中提供的固定旋转。`T_align` 错误会导致末端姿态出现系统性偏差。

---

## 推理流程

### Episode 开始时执行一次

1. **估计物体位姿**：`Perception.estimate_objects()` 检测并分割每个物体，将掩膜像素通过深度提升到三维空间，然后拟合 6DoF 位姿。锚点物体 `obj1` 定义 ICT 使用的物体中心参考系。
2. **机械臂回到 Home 位并张开夹爪**。

### 每个闭环控制周期执行（约 5～10 Hz）

| 步骤 | 操作 | 代码位置 |
|---|---|---|
| 3 | 获取同步 RGB-D 图像 | `Camera.get_frame()` |
| 4 | 读取每台机械臂的末端 FK 位姿和夹爪状态，并通过 `T_align` 转成手位姿 | `RobotArm.get_T_ee_in_cam()` |
| 5 | 锁定被抓取物体，使其估计位姿随夹爪运动 | `run_inference.py: latch_objects()` |
| 6 | 构造干净图像：移除真实机械臂并绘制虚拟夹爪 | `Perception.make_clean_image()` |
| 7 | 根据手和物体位姿构造 ICT | `ICTPolicy.build_ict()` |
| 8 | 执行流匹配推理，得到未来末端轨迹和完成概率 | `ICTPolicy.infer()` |
| 9 | 将参考系下的预测解码为相机坐标系末端目标 | `ICTPolicy.decode_ee_in_cam()` |
| 10 | 平滑并执行前 `exec_horizon` 步，然后重新规划 | `TrajectoryController.execute_chunk()` |
| 11 | 当 `done_prob > done_threshold` 时结束 episode | `run_inference.py: run()` |

### ICT 的构造方式

`ICTPolicy.build_ict()` 与 `training/FlowMatchingDataloader._build_ict()` 保持一致。每个实体（手或物体）对应一个 Token：

```text
[ type_id(1) | pose_in_ref(9) | hand-in-entity(单臂 9 / 双臂 18) | flag(1) ]
```

- `pose_in_ref`：实体在参考坐标系中的 6DoF 位姿，编码为 `[归一化位置(3), 6D旋转(6)]`。
- `hand-in-entity`：每只手在当前实体坐标系中的相对位姿。这种相对编码使表示以交互为中心，并降低相机和机器人位置变化的影响。
- Token 顺序固定为：**手在前，然后是锚点物体，最后是其余物体**。该顺序必须与训练阶段一致。

### 机械臂如何执行策略输出

策略预测参考坐标系下未来的“手”轨迹。每一步通过 `decode_ee_in_cam()` 转回机械臂末端目标：

```text
预测手位姿（REF）--T_ref_in_cam--> 手位姿（CAM）--inv(T_align)--> 末端位姿（CAM）
```

`TrajectoryController` 对位置进行 EMA 平滑，对旋转进行 Slerp 平滑，并限制每个周期的位置和旋转步长。随后它以非阻塞方式调用 `RobotArm.move_ee_in_cam(...)`，同时根据预测抓取概率控制夹爪开合。每次只执行预测轨迹前 `exec_horizon` 步，然后根据新观测重新规划。这种**滚动时域控制**使策略能够持续响应环境变化。

---

## 真实机器人运行方法

### 前置条件

1. 已训练的 checkpoint，且同目录中包含 `config.json` 和 `dataset_stats.json`。二者由训练器生成，分别记录模型结构和归一化统计量。
2. 已安装硬件驱动：`SKIP_HARDWARE=0 bash setup.sh`（RealSense + Trossen）。
3. 已准备感知模型。参考实现使用 `preprocess/` 中的 DINO-SAM 和 LaMa。
4. 已完成每台机械臂的手眼标定，并得到 `T_base_in_cam`。

### 配置

编辑 [`../cfg/inference/example_dualarm.yaml`](../cfg/inference/example_dualarm.yaml)，设置：

- `policy.ckpt`
- `perception.object_prompts` 和 `erase_prompt`
- 相机及机械臂的 `cfg_path`
- `robot.T_align`

### 启动

```bash
python inference/run_inference.py cfg/inference/example_dualarm.yaml
```

---

## 配置与调参

主要参数均位于 `example_dualarm.yaml`：

| 参数 | 配置段 | 作用 |
|---|---|---|
| `num_inference_steps` | `policy` | 流 ODE 求解步数，通常为 10～20；越大通常越平滑，但推理更慢。 |
| `exec_horizon` | `control` | 每次重新规划前执行的预测步数；越小越灵敏，但推理频率更高。 |
| `control_hz` | `control` | 控制循环频率，同时决定 `dt`；应与机械臂伺服频率匹配。 |
| `alpha_pos` / `alpha_rot` | `control` | 位置 EMA 和旋转 Slerp 平滑系数；越大越平滑，但延迟更明显。 |
| `max_pos_step` | `control` | 每个周期允许的最大末端位移，单位为米。初次上机应设置得较小。 |
| `max_rot_step` | `control` | 可选的单周期最大旋转角，单位为弧度。 |
| `grasp_threshold` | `control` | 预测抓取概率超过该阈值时闭合夹爪。 |
| `done_threshold` | `control` | 完成概率超过该阈值时结束 episode。 |
| `safe_z_min` | `control` | 预期的基座坐标系 Z 轴下限，用于保护桌面。当前真实机器人模板尚未将该值传入通用驱动，Trossen 示例驱动内部使用 −0.12 m 下限，因此仍应依赖驱动层安全限制和急停。 |

首次上机时，建议使用较低的 `control_hz`、较小的 `max_pos_step` 和较高的 `safe_z_min`，并随时准备触发急停。确认运动方向和姿态正确后再逐步放宽限制。

---

## 单臂与双臂

模板默认使用双臂：

```yaml
robot.sides: ["left", "right"]
```

对应训练配置中的 `single_hand: false`，此时 `ict_dim = 29`，一次前向推理同时预测左右手轨迹。

使用单臂时，应设置：

```yaml
robot.sides: ["right"]
```

并使用以 `single_hand: true` 训练的 checkpoint，此时 `ict_dim = 20`。`ICTPolicy` 会从 checkpoint 同目录的 `config.json` 中读取 `single_hand`，自动选择正确的 Token 布局和轨迹解析方式。

---

## 接入自己的相机、机械臂和感知模块

只需实现 `interfaces.py` 中的三个接口：

- **`Camera`**：返回 `Frame(rgb, depth_m, K)`。可参考 `run_inference.py` 中封装 `CamRS` 的 `RealSenseCamera`。
- **`RobotArm`**：实现 FK（`get_T_ee_in_cam`）、笛卡尔伺服（`move_ee_in_cam`）、夹爪控制、`go_home`、资源释放，并提供 `T_base_in_cam`。可参考封装 `RobotArmTrossen` 的 `TrossenArm`。笛卡尔 IK 由机械臂驱动负责。
- **`Perception`**：返回物体 6DoF 位姿和干净图像。这通常是移植工作中最复杂的部分。物体位姿可来自开放词汇检测器与深度 PCA、AprilTag、FoundationPose、已知 CAD 模型与 ICP，或其他来源。干净图像必须与模型训练阶段使用的图像表示保持一致，包括图像修复和虚拟夹爪渲染方式。

---

## 参考模板未包含的生产功能

为便于理解，本模板省略了内部生产推理循环中的部分功能：

- **异步控制和时序集成**：由独立线程以固定频率执行伺服，与较慢的策略推理解耦，并对重叠预测进行加权平均以提高平滑性。
- **Delta 动作模式、点云特征、区域注意力、物体动力学和视觉预测辅助头**：模型层支持这些能力，但此模板主要展示常用的绝对动作路径。
- **完整的鲁棒性与交互功能**：键盘遥控接管、实时可视化、抓取后强制抬升、IK 失败逃逸、交互式外参标定、遮挡期间持续抓取锁定，以及 checkpoint 结构自动识别。

建议先让单臂可靠地到达物体，再逐步增加这些功能。

---

## MuJoCo `serve_bread` 验证

当前工作区包含针对发布版单右手 `serve_bread` checkpoint 的 MuJoCo 适配实现：

- `mujoco_backend.py` 持有共享的模型和仿真数据，负责渲染、物理步进以及 OpenCV 相机坐标转换。
- `mujoco_adapters.py` 实现 `Camera`、每周期读取真值的 `Perception`，以及带安全约束 DLS IK 的 Piper `RobotArm`。
- `run_mujoco_inference.py` 支持开环轨迹诊断和闭环控制。
- [`../cfg/inference/mujoco_serve_bread.yaml`](../cfg/inference/mujoco_serve_bread.yaml) 包含场景、IK、控制和安全参数。

### 环境准备

```bash
conda activate humanego
cd /home/yyn/workspace/HumanEgo
```

### 开环检查

首先只运行策略并保存预测轨迹，不移动机械臂：

```bash
MUJOCO_GL=egl python inference/run_mujoco_inference.py --dry-run --cycles 1
```

### 闭环运行

无界面运行：

```bash
MUJOCO_GL=egl python inference/run_mujoco_inference.py --cycles 250
```

打开 MuJoCo Viewer：

```bash
python inference/run_mujoco_inference.py --viewer
```

诊断图像保存在 `runs/mujoco_serve_bread/`。图中的紫色曲线表示策略预测的 50 步末端轨迹。

仿真推理与真实机器人模板有三点区别：

1. 机械臂 geom 位于独立渲染组，生成策略图像时直接隐藏，无需 LaMa 图像修复。
2. 每个控制周期直接读取 MuJoCo 中牛角包和盘子的真实位姿，无需 DINO-SAM；物理牛角包在双指接触后跟随模型手部运动，同时保留发布 checkpoint 训练时的静态 `obj1` anchor 输入。
3. HumanEgo 输出的相机坐标系 6DoF 目标经过坐标变换和 DLS IK 后写入 Piper 位置执行器；仿真后端在每个控制周期推进对应数量的物理步。

默认关闭墙钟实时等待（`realtime_sleep: false`），因此仿真验证会以更快速度运行；MuJoCo 的控制时间步和模型轨迹执行步数不变。终端日志中的 `gseq` 是当前预测 chunk 的夹爪概率，`plateA`/`handA` 是放置阶段的 anchor-frame 输入诊断。

为避免把侧面碰撞误判为成功抓取，Piper 适配器还会直接读取 MuJoCo
接触信息。策略要求闭合时，夹爪必须先满足距离门槛；闭合过程中暂停
笛卡尔运动，只有左右两指都接触面包（日志为 `contacts=11`）后才进入
`grasped` 并恢复搬运。常见物理抓取状态如下：

| `physical` | 含义 |
| --- | --- |
| `open` | 夹爪张开，未请求抓取。 |
| `waiting_near_object` | 策略要求闭合，但夹爪离面包仍过远。 |
| `closing` | 夹爪正在原地闭合，机械臂暂停运动。 |
| `grasped` | 左右两指均与面包接触，可以搬运。 |
| `aligning` | 单指接触后夹爪保持张开，依次退开、横向对中并重新接近物体。日志中的 `align` 显示具体子阶段。 |
| `alignment_required` | 只有一根手指接触；保持张开并停止运动，防止侧推或拖动。 |
| `retry_wait` | 一次闭合未形成双指接触，夹爪重新张开等待。 |

日志中的 `ee_obj` 是配置抓取参考点到面包中心的距离（米），`contacts=10`
或 `01` 表示单指接触，`contacts=11` 才表示经过物理验证的双指夹持。
自动对中由 `alignment_retreat_distance`、`alignment_max_step`、
`alignment_damping`、`alignment_max_joint_step` 和 `alignment_timeout` 控制；
对中阶段以位置修正为主，并用较小旋转权重保持首次接触姿态。
在尚未接触且 `ee_obj` 小于 `prealign_distance` 时，适配器会先动态修正
策略目标沿夹爪闭合轴的分量；这不会修改 ICT 或 `T_align`，策略仍控制
末端姿态。日志中的 `prealign=orienting` 表示保持位置等待抓取姿态，
`prealign=centering` 表示保持姿态进行横向对中。

当前 `serve_bread` 配置默认将 `auto_align_single_contact` 设为 `false`：
单指接触进入 `alignment_required` 并保持张开，优先保证不会误闭合或拖动物体。
完成具体场景的基座位置和工具偏置标定后，再启用自动对中。

建议先运行 `--dry-run` 检查紫色预测轨迹和坐标方向，再启动闭环控制。当前仿真配置启用了工作空间限制、位置/旋转步长限制、IK 远分支拒绝和关节命令步长限制。
