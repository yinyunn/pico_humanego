# HumanEgo — 训练

基于预处理后的 Aria 数据训练 HumanEgo **流匹配策略（flow-matching policy）**。该策略根据一张第一视角图像和一组 *Interaction-Centric Tokens（ICTs，交互中心标记）*——场景中的手和物体——预测未来一小段时间内手部（以及物体）的 6-DoF 运动。本说明涵盖：（1）如何训练；（2）所需的数据格式；（3）`training/` 中的文件；（4）一次运行会生成的内容；（5）所有配置参数以及如何添加自定义任务。

---

## 1. 快速开始

安装环境（在仓库根目录执行 `bash setup.sh`），准备一些**预处理后的**数据——可以下载已发布的任务，或者对自己的录制数据运行[预处理](../preprocess/README.md)——然后执行：

```bash
# serve_bread，已发布的 HumanEgo 训练配置
python -m training.FlowMatchingTrainer --task serve_bread --use_cfg --job HumanEgo
```

`--task` 用于选择数据和配置目录，`--job` 用于选择 YAML 配置文件（`cfg/training/serve_bread/HumanEgo.yaml`）。输出会保存到 `runs/serve_bread/HumanEgo/`。

---

## 2. 所需数据

训练使用**预处理步骤**（见[预处理](../preprocess/README.md)）生成的输出，每段录制数据对应一个目录：

```
data/<task>/aria/
└── mps_<task>_<id>_vrs/
    └── preprocess/
        └── all_data/
            ├── 00000/   training_data.json + rgb*.png / mask*.png
            ├── 00001/   ...
            └── ...
```

每个 `training_data.json` 都是由预处理步骤写入的逐帧目标数据（包括相机、手和物体的 SE(3) 位姿、抓取状态以及图像路径——详见[预处理输出参考](../preprocess/README.md#44-training_datajson-schema)）。数据加载器会收集所有训练会话中的 `all_data/<idx>/training_data.json`，并读取由 `img_name` 指定的图像变体。

**训练集 / 评估集划分（`data_sources` 模式）。** 当使用 `data_sources: {aria: N}` 时，训练器会自动发现 `data/<task>/aria/mps_<task>_*_vrs`，将录制数据 `000` **留作评估**，并使用接下来的 `N` 段录制数据进行训练。因此至少需要 **2 段录制数据**（1 段评估数据 + 1 段训练数据）；`eval_source` 用于选择提供留出会话的来源。也可以改为显式传入 `--train_data` / `--eval_data` 会话列表。

---

## 3. `training/` 中的文件

| 文件 | 说明 |
|------|------|
| `FlowMatchingTrainer.py` | **入口文件**——负责命令行参数、配置解析、训练 / 评估循环和检查点保存。使用 `python -m training.FlowMatchingTrainer` 运行。 |
| `FlowMatchingModel.py` | **策略网络**——基于 ICTs 的流匹配解码器，可选区域感知注意力、点云注入以及辅助协同训练头。 |
| `FlowMatchingDataloader.py` | 根据 `training_data.json` 构建逐帧**样本**：图像、ICTs（手和物体）以及未来时间范围内的目标。实现范式消融实验（frame / centric / action 模式）、双手模式和数据增强。 |
| `FlowMatchingEvaluator.py` | 教师强制（teacher-forced）的**可视化评估器**——在训练期间渲染真实轨迹与预测轨迹对比视频。 |

---

## 4. 一次运行会生成的内容

所有内容都会保存到 `runs/<task>/<job>/`（使用 `--exp` 时保存到 `runs/<task>/<exp>/<job>/`）：

| 文件 | 含义 |
|------|------|
| `latest.pt` | 检查点（模型、优化器和 epoch）。如果该文件存在，训练会**自动从此处恢复**。 |
| `dataset_stats.json` | 归一化统计信息——只计算一次并缓存。 |
| `config.json` | 本次运行实际使用的完整配置。 |
| `train_curve.png`、`eval_curve.png` | 按 epoch 绘制的训练 / 评估损失曲线。 |
| `eval_snapshots/eval_ep_*.json` | 每个 epoch 的评估指标。 |
| `eval_render/epoch_*/` | 真实轨迹与预测轨迹对比视频（每隔 `vis_eval_every` 个 epoch 生成一次）。 |

---

## 5. 配置

### 5.1 配置的解析方式

训练器首先使用 `FlowMatchingTrainer.py` 顶部 `TrainConfig` 中的默认值，然后按以下顺序应用配置：

1. **YAML**——使用 `--use_cfg` 时，加载 `cfg/training/<task>/<job>.yaml`（指定 `--exp` 时加载 `cfg/training/<task>/<exp>/<job>.yaml`）。
2. **CLI 参数**——命令行中传入的参数（例如 `--epochs 200 --lr 5e-5`）会覆盖 YAML 中的对应值。大多数 `TrainConfig` 字段都有对应的命令行参数；数据源字段（`data_sources`、`data_root`、`eval_source`）在 YAML 中设置，而不是通过命令行设置。

运行目录始终为 `runs/<task>/<job>/`，同时 `--task` 也决定要加载的数据位置（`data/<task>/...`）。

```bash
python -m training.FlowMatchingTrainer --task <task> --use_cfg --job <job> [--exp <group>] \
    [--epochs N] [--lr 1e-4] [--data_num K] ...
```

### 5.2 参数参考

> 以下默认值是 `TrainConfig` 的回退值。已发布的 `HumanEgo.yaml` 会覆盖其中一些配置（损失权重、范式标志、AMP 等）——**该文件才是标准训练方案**；建议复制它，而不是从零开始创建配置。

**数据与划分**

| 配置项 | 默认值 | 说明 |
|-----|---------|---------|
| `data_sources` | `null` | `{aria: N}`——按数据源自动发现会话；将 `000` 留作评估，并使用接下来的 N 段数据训练。 |
| `eval_source` | `aria` | 提供留出评估会话的数据源。 |
| `data_num` | `null` | 训练会话数量上限（在数据划分后应用）。 |
| `task` | `serve_bread` | 数据和配置目录；由 `--task` 设置。 |
| `data_root` | `./data` | 数据集根目录。 |

**优化与训练计划**

| 配置项 | 默认值 | 说明 |
|-----|---------|---------|
| `epochs` | 400 | 训练轮数。 |
| `batch_size` | 32 | 批大小。 |
| `lr` | 1e-4 | AdamW 学习率。 |
| `weight_decay` | 0.01 | AdamW 权重衰减。 |
| `grad_clip` | 1.0 | 梯度范数裁剪值。 |
| `use_lr_schedule` | False | 使用带预热的余弦学习率调度。 |
| `warmup_steps` / `min_lr_ratio` | 200 / 0.05 | 预热步数；学习率下限占 `lr` 的比例。 |
| `use_amp` | False | 混合精度（AMP）训练。 |
| `use_ema` / `ema_decay` | True / 0.999 | 维护一份模型权重的指数移动平均副本。 |

**策略输入、输出与预测范围**

| 配置项 | 默认值 | 说明 |
|-----|---------|---------|
| `pred_horizon` | 50 | 策略预测的未来步数（动作块长度）。 |
| `image_size` | [240, 320] | 输入图像尺寸（H、W）。 |
| `img_name` | `rgb_WoArm_WArmObjKpts.png` | 要输入的预处理图像变体；设为 `None` 表示仅使用状态输入（不使用视觉）。 |
| `single_hand` / `single_hand_side` | False / "right" | 单手或双手模式；单手模式下指定使用哪只手。 |
| `max_ict` | 8 | ICT（手和物体）的最大数量。 |
| `hand_tracking_method` | `aria_mps` | 从 `training_data.json` 读取的手部数据来源。 |

**范式（模型归纳偏置）**

| 配置项 | 默认值 | 说明 |
|-----|---------|---------|
| `centric_mode` | `object_centric` | 参考坐标系原点：以物体为中心（`object_centric`）或以自我为中心（`ego_centric`）。 |
| `frame_mode` | `anchor_frame` | 相对于锚定物体（`anchor_frame`）或相机坐标系（`camera_frame`）进行预测。 |
| `action_mode` | `absolute` | 预测绝对位姿或增量（`delta`）步骤。 |
| `use_region_attn` | False | 可学习的区域感知（“聚光灯”）注意力偏置。 |
| `use_pcd_features` | False | 注入显式的三维点云特征。 |
| `use_ot_cfm` | False | 最优传输条件流匹配（生成更直的流）。 |

**辅助协同训练**（每项都会增加一个预测头和一个损失项）

| 配置项 | 默认值 | 说明 |
|-----|---------|---------|
| `use_aux_obj_dynamics` | False | 联合建模物体运动。 |
| `use_aux_visual_foresight` | False | 预测未来二维空间热力图。 |
| `use_aux_temporal_contrastive` | False | 在潜空间中预测未来 ICTs。 |

**损失权重**

| 配置项 | 默认值 | 说明 |
|-----|---------|---------|
| `w_flow` | 3.0 | 流匹配速度损失。 |
| `w_pos` / `w_rot` | 2.0 / 1.0 | 手部位置 / 旋转损失。 |
| `w_g` | 10.0 | 抓取损失。 |
| `w_done` | 5.0 | 完成标志损失。 |
| `w_foresight` / `w_contrastive` | 1.0 / 1.0 | 上述两个辅助预测头的权重。 |

**模型架构**

| 配置项 | 默认值 | 说明 |
|-----|---------|---------|
| `patch_size` | 16 | 视觉 patch 尺寸。 |
| `vision_embed_dim` | 384 | 视觉 / token 嵌入维度。 |
| `num_decoder_layers` / `num_heads` | 6 / 8 | Transformer 解码器深度 / 注意力头数。 |
| `mlp_ratio` / `dropout` | 4.0 / 0.05 | MLP 扩展比例；dropout 比例。 |

**流匹配推理与评估**

| 配置项 | 默认值 | 说明 |
|-----|---------|---------|
| `num_inference_steps` | 10 | 采样时的流积分步数。 |
| `eval_every` / `vis_eval_every` | 1 / 50 | 每隔 N 个 epoch 执行评估 / 生成评估视频。 |

**数据增强**——`enable_augmentation` 是总开关，另有以下各类型开关：`enable_aug_img`、`enable_aug_rrc`（随机缩放裁剪）、`enable_aug_target_jittering`、`enable_aug_cutout`、`enable_aug_temporal_stride`、`enable_aug_interpolation`。

**旧版兼容性**（用于匹配已发布的训练方案；除非确定要修改，否则保持 `HumanEgo.yaml` 中的设置）：`use_pre_norm`、`use_ctx_norm`、`use_done_in_flow`、`use_legacy_image_loading`、`use_legacy_rng`。

如需查看精确默认值，请阅读 `FlowMatchingTrainer.py` 中的 `TrainConfig`。

### 5.3 添加自定义配置

要在**自己的任务**上训练策略：

1. **预处理**录制数据（见[预处理](../preprocess/README.md)），确保拥有 `data/<your_task>/aria/mps_<your_task>_*_vrs/preprocess/all_data/…`。至少需要 **≥2 段录制数据**（其中 1 段留作评估）。
2. **创建** `cfg/training/<your_task>/HumanEgo.yaml`——最简单的方式是复制 `cfg/training/serve_bread/HumanEgo.yaml`，然后调整：
   - `data_sources: {aria: N}`——将 `N` 设为训练录制数据的数量。
   - `single_hand` / `single_hand_side`——单手任务设为 `True` / `"right"`，双手任务设为 `False`。
   - 只有在任务确实需要时才修改损失权重 / 范式标志；否则保留已发布的默认配置。
3. **训练：**
   ```bash
   python -m training.FlowMatchingTrainer --task <your_task> --use_cfg --job HumanEgo
   ```
4. **查看** `runs/<your_task>/HumanEgo/`——`eval_curve.png` 和 `eval_render/epoch_*/` 中的视频可以展示训练进度；如果训练被中断，会从 `latest.pt` 自动恢复。

> 冒烟测试：在进行完整训练前，添加 `--epochs 5 --data_num 1`，确认数据可以正常加载且至少能完成一个训练步骤。
