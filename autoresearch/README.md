# AutoHumanEgo V1 基础设施

这是当前 HumanEgo 仓库的配置级 AutoResearch 控制面。它与受保护的 HumanEgo
训练、预处理和推理实现保持独立。

## 针对当前仓库的适配

- 任务：`open_door`
- 原始基线：`cfg/training/open_door/OpenDoorLeftHandOnly.yaml`
- 基线副本：`cfg/training/open_door/autoresearch/exp_0000.yaml`
- 训练器：`python -m training.FlowMatchingTrainer`
- 配置路径：`cfg/training/<task>/<exp>/<job>.yaml`
- 运行路径：`runs/<task>/<exp>/<job>/`
- 基线命令：见 `autoresearch/logs/exp_0000.md`
- 数据划分：当前 open_door 数据来自仓库外的
  `/home/yyn/workspace/pico_download/open_door/humanego_data/`。基线配置显式
  指定 1 个评估 session（`20260827_212341`）和 29 个训练 session
  （从 `20260827_212421` 到 `20260827_213522`），不依赖仓库内为空的
  `data/open_door/` 自动发现逻辑。

指南建议的 `exp_0000.yaml` 布局和命令已适配为训练器实际支持的
`--exp autoresearch --job exp_0000` 接口。未向受保护源码添加兼容性补丁。

## 官方评估与分数

现有的 `training/FlowMatchingTrainer.py` 会在
`eval_snapshots/eval_ep_*.json` 中计算无教师 ODE 指标：

```text
score = pos_err_w_m + rot_err_w_deg / 100.0 - grasp_f1_w * 0.10
```

分数越低越好。`evaluate_experiment.py` 只负责解析和验证这些快照，不会替换
或修改 HumanEgo 评估器。

## 当前状态

当前已从 `config_search` 切换到 `open_door` 的 `loss_search`。已完成恰好
`exp_0007`–`exp_0016` 共 10 个实验；本批次只研究
`w_flow`、`w_pos`、`w_rot`、`w_g`、`w_done`、`w_foresight`、
`w_contrastive`。除这些损失权重外的训练超参数、数据、验证划分和模型架构均冻结。
当前最佳为 `exp_0014`；自主循环已关闭，未经用户授权不得启动 `exp_0017`。

## 工具脚本

- `scripts/validate_config.py`：根据当前仓库实际使用的 `TrainConfig` 字段验证 YAML。
- `scripts/run_experiment.py`：启动一个明确指定且相互隔离的训练任务，不包含自主循环。
- `scripts/evaluate_experiment.py`：解析单个实验最新的评估 JSON。
- `scripts/leaderboard.py`：按分数排列已记录的成功实验。
- `scripts/verify_repo_clean.py`：在未来实验前报告受保护文件的改动。
