# PICO 4 Ultra 移植审计

本审计记录了新增后端前检查过的接口。PICO 后端不会创建 `.vrs`、MPS CSV
或 Project Aria 兼容文件。

## 实际可用的接口

### HumanEgo

- `preprocess/AriaCam.py` 生成逐帧 `c2w`、`k`、`d`、RGB 和设备时间戳，
  并明确使用 `T_c2w = T_d2w @ T_c2d` 进行组合。
- `preprocess/AriaHands.py` 最终向 `DatasetGen.py` 提供世界坐标系下的手部
  变换和抓取状态；其内部的 21 点表示属于 Aria 专用格式，不用于 PICO 原始数据。
- `preprocess/AriaSlam.py` 根据相机/世界位姿和时间戳计算速度，`AriaPhases.py`
  消费这些运动学数据。PICO 后端使用带来源信息的文件名实现相同的下游字段。
- `preprocess/DatasetGen.py` 当前仍只服务于原有 Aria 阶段文件；PICO 原始数据
  暂不写入该下游目录。

### XRoboToolkit PC Service / Python binding

检查过的 Python README 暴露了头显位姿、设备时间戳、手部追踪状态和
`isActive`。检查过的 binding 源码表明，其回调已经收到完整的
`HandJointLocations[26]`，但旧 Python API 丢弃了质量字段，只返回固定的
26×7 数组。

本次 binding 的最小修改新增：

- `get_left_hand_tracking_json()`
- `get_right_hand_tracking_json()`

这两个接口保留 SDK 原始手部对象，包括 `p`、`s`、`r`、`isActive`、`scale`
和 `timeStampNs`。旧接口仍然可用。修改后的本地 binding 已成功重建，
但本次实现没有连接真实 PC Service/PICO 设备。

检查过的 binding/README 没有暴露 VST 视频流、相机内参或头显-相机外参 API。
因此录制器要求输入带原始时间戳的 RGB 目录和经过验证的标定文件；缺少这些
数据时会明确失败。

SDK README 将头显坐标系描述为右手系 X=右、Y=上、Z=向内，原点为应用启动时的
头显位置；手部状态位为 OrientationValid=`0x1`、PositionValid=`0x2`、
OrientationTracked=`0x4` 和 PositionTracked=`0x8`。同一 README 对序列化的
关节 pose 使用了“左手系”标签，但仍描述 Z 轴向内。由于这一点存在歧义，
实现将 `pico_to_humanego_axes` 显式保存在 session 中，并要求真实相机外参，
而不是静默假定一个固定的坐标转换。

## 复用与重写范围

| 模块 | 处理方式 |
| --- | --- |
| Aria 数据采集 | 保持不变 |
| RGB/追踪时间戳契约 | `tools/pico_import/` 生成 source-native 同步 manifest |
| 位姿计算与时间同步 | 当前只做时间同步；坐标适配待同步验收后实现 |
| HumanEgo Stage 1 | 暂不接入 |
| DINO/SAM2、CoTracker、CamTriangulator、LaMa、DatasetGen | 保持原有 Aria 流程 |
| VST 采集 | 不伪造；需要经过验证的 Unity/PICO 采集链路 |

## 硬件阻塞项

当前同步层要求输入视频、视频 PTS 和原始 trackingData；没有三者时不会生成
同步 manifest。同步通过后，再单独设计 HumanEgo 下游接入层。
