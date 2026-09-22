# PICO v3 Milestone 2：双目几何诊断

日期：2026-09-09。状态：`BLOCKED_CALIBRATION_REVIEW`。

本轮实现了同一解码帧的 MediaPipe 左右对应、线性三角化、正深度、ray angle、
极线误差、左右重投影误差、21 点完整性、骨长/手尺寸检查，以及复用现有
midpoint frame 数学定义的诊断 gripper。所有失败结果均标记无效；没有用
MediaPipe z、跨时间 SLAM 或 native 26 点补洞。

## 用户确认的几何定义

本轮只保留以下定义：

```text
E: raw camera -> head
D: raw camera -> image camera
p_head = E @ inv(D) @ p_image
T_right_image_from_left_image
  = inv(E_right @ inv(D)) @ (E_left @ inv(D))
```

`D` 是坐标基转换，不是左右相机安装位姿。代码不会用 `inv(E)` 取代已确认的
方向，也不会根据 MediaPipe 或背景点拟合外参。早期诊断目录
`outputs/pico_v3_geometry_audit/recording_*` 中的 inverse-header 实验已废弃；
有效输出只看 `outputs/pico_v3_geometry_audit/confirmed_*`。

## 修改范围

- 新增 `tools/pico_import/stereo_geometry.py`：纯数学 helper，没有新 adapter/backend。
- 修改现有 `humanego_writer.py`：增加只读 `--stereo-3d-diagnostic` 入口；
  原 native export 默认路径未改。
- 修改现有 `visuals.py`：生成极线、native/MediaPipe/重投影和诊断 gripper 视频与 contact sheet。
- 新增 `tests/test_pico_stereo_geometry.py`：9 项合成 metric geometry 测试。
- `calibration.py`、tracking/head parser、CoTracker、CamTriangulator、DatasetGen、phase 和训练代码均未改。

## 真实数据结果

| 录制 | 帧数 | 观测右减左视差中位数 `(du,dv)` | 已确认几何极线误差中位数 | 有效 joint 比例 | 通过整手数 |
|---|---:|---:|---:|---:|---:|
| Open Door `20260907_174903` | 423 | `(-86.38,-0.34) px` | `86.38 px` | `0.0%` | 0 |
| Desk `20260908_174122` | 765 | `(-135.86,-0.61) px` | `139.48 px` | `0.0%` | 0 |

header 推导的基线长度为 `0.064013 m`。按已确认 `E+D` 定义转换到 image-camera
坐标后，基线主要落在 image Y 方向；两段 RenderMode_3D 视频中的对应点主要呈
image X 方向视差。合成数据的正确/错误基线、正负视差、非平行相机和退化点测试
均通过，因此当前失败不是 `cv2.triangulatePoints` 的基本公式错误。

录制端源码确认 MP4 通过 `PXRCapture_RenderMode_3D` 生成，header 同时调用
`getCameraExtrinsics()`。现有 metadata 没有说明 RenderMode_3D 半幅是否仍属于
raw-camera 外参对应的投影域，也没有 distortion/rectification 参数。
目前不能判断缺的是 RenderMode 的图像变换、每目参数，还是录制端额外标定。

## 给人工检查的文件

Open Door：

- `outputs/pico_v3_geometry_audit/confirmed_20260907_174903/stereo_3d_diagnostic/epipolar_geometry_contact_sheet.jpg`
- 同目录 `stereo_reprojection_contact_sheet.jpg`
- 同目录 `epipolar_geometry.mp4` 和 `stereo_reprojection.mp4`
- 同目录 `summary.json` 和 `stereo_3d.jsonl`

Desk 使用对应目录 `confirmed_20260908_174122/stereo_3d_diagnostic/`。

极线图中，同色元素依次表示左目 landmark、右目实际 landmark、由左目点和
当前几何推导的右目极线；点应落在线上。重投影图中绿色是 MediaPipe 2D，红色
圆圈是三角化重投影；两者应重合。画面还叠加 native skeleton 供比较。

## 验证与下一步

- 9 项 `unittest` 合成几何测试通过。
- 原 PICO import 的 8 项测试函数通过。
- 两段输出 MP4 均可解码到完整 423/765 帧，JSONL 与实际 PTS 已在 Milestone 1 检查。
- final `aria_hands.json` 仍来自 native PICO，`hand_source` 没有伪装成 mediapipe_stereo。
- phase 仍未修改，符合先验证 hand 再做 phase 的顺序。

### 显式 image-camera 重算与 RenderMode 候选

根据人工观察到的“右目关键点不在极线上、右目重投影向右下大幅偏移”，三角化
已改为直接构造：

```text
P_left  = K @ D @ inv(E_left)
P_right = K @ D @ inv(E_right)
X_head  = triangulate(P_left, P_right, uv_left, uv_right)
X_world = HeadPose @ X_head
```

质量门控也完全使用这一 head-frame DLT 解，不再借用另一坐标原点下的失败解。
新增合成测试证明 `D` 被显式作用到 raw-camera 输出，并验证完美观测时与
image-camera 相对位姿形式等价。真实数据上显式公式仍失败，说明问题不只是漏乘 D。

随后加入 `render_mode_3d_rectified_candidate`：左目到 head/world 仍使用确认的
`E_left @ inv(D)`；仅将 MP4 的 RenderMode_3D 左右半幅视作水平校正的
image-camera 对。相对旋转取单位阵，基线长度取 header 两相机中心距离
`0.064013 m`，符号由左右图像顺序确定。它不修改 PICO 外参，也不是新标定文件。

Open Door 423 帧的对比如下：

| 投影模型 | 极线误差中位数 | joint 有效率 | 左/右重投影中位数 | 完整手通过数 |
|---|---:|---:|---:|---:|
| 确认的 `K D inv(E)` | 86.38 px | 0.00% | 1469.72 / 1464.53 px | 0 |
| RenderMode_3D 水平候选 | 1.71 px | 90.94% | 0.854 / 0.854 px | 163 |

最新人工检查文件位于：

- `outputs/pico_v3_geometry_audit/render_mode_candidate_20260907_174903/stereo_3d_diagnostic/epipolar_geometry_contact_sheet.jpg`
- 同目录 `stereo_reprojection_contact_sheet.jpg`
- 同目录 `epipolar_geometry.mp4` 和 `stereo_reprojection.mp4`

contact sheet/video 的上半部分是确认的 `K D inv(E)`，下半部分是 RenderMode 候选。
下半部分同色右目点应落在极线上，红色重投影应覆盖绿色 MediaPipe landmark，
并在通过整手质量检查时显示诊断 gripper 坐标轴。

RenderMode 候选通过像素几何不等于已证明 world pose 正确。继续 final hand source
前仍需人工确认下半部分；更严谨的最终依据是 RenderMode_3D 对应的每目投影参数，
或与该 MP4 像素域一致的独立双目标定。
