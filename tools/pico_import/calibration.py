"""从 PICO tracking header 解析左右目适配所需的标定参数。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PICO4_ULTRA_KA = np.array(
    [[686.8340695363, 0.0, 545.5], [0.0, 686.8680648828, 384.5], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
PICO4_ULTRA_CAMERA_TO_IMAGE_D = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
PICO4_ULTRA_RIGHT_EXTRINSIC_INDEX = 0
PICO4_ULTRA_LEFT_EXTRINSIC_INDEX = 1
PICO4_ULTRA_EXTRINSIC_INDEX = {
    "right": PICO4_ULTRA_RIGHT_EXTRINSIC_INDEX,
    "left": PICO4_ULTRA_LEFT_EXTRINSIC_INDEX,
}


def _validate_eye(eye: str) -> str:
    if eye not in PICO4_ULTRA_EXTRINSIC_INDEX:
        raise ValueError("eye must be left or right")
    return eye


def _matrix_pair(value) -> list[np.ndarray]:
    if isinstance(value, str):
        parts = value.split("|")
        return [np.asarray(json.loads(part), dtype=np.float64).reshape(4, 4) for part in parts]
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (2, 4, 4):
        return [array[index] for index in range(2)]
    if array.size == 32:
        return [item.reshape(4, 4) for item in array.reshape(2, 4, 4)]
    raise ValueError("cameraExtrinsics must contain two 4x4 matrices")


def _numeric_tuple(value, name: str) -> tuple[float, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    result = tuple(float(item) for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _load_header(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if "notice" in value:
            return value
    raise ValueError(f"PICO tracking header not found: {path}")


@dataclass(frozen=True)
class PicoCalibration:
    """同步阶段使用的单目标定快照。

    PICO header 的 E0 对应右目，E1 对应左目。``E_eye`` 的方向按
    ``T_camera_to_head`` 使用；``D`` 只负责 raw camera basis 到 image camera
    basis 的转换，左右目共用同一坐标轴变换。
    """

    K: np.ndarray
    E_left: np.ndarray
    E_pair: tuple[np.ndarray, np.ndarray]
    D: np.ndarray
    raw_intrinsics: tuple[float, ...]
    left_extrinsic_index: int = PICO4_ULTRA_LEFT_EXTRINSIC_INDEX
    eye: str = "left"

    def __post_init__(self):
        _validate_eye(self.eye)

    @property
    def extrinsic_index(self) -> int:
        return PICO4_ULTRA_EXTRINSIC_INDEX[self.eye]

    @property
    def E_eye(self) -> np.ndarray:
        """当前选定目对应的 header 外参。"""
        return self.E_pair[self.extrinsic_index]

    @property
    def E_right(self) -> np.ndarray:
        """header E0（右目）外参。"""
        return self.E_pair[PICO4_ULTRA_RIGHT_EXTRINSIC_INDEX]

    @property
    def camera_profile(self) -> str:
        return f"pico4_ultra_{self.eye}_e{self.extrinsic_index}_head_direct_handworld_axes_d"

    @classmethod
    def from_header(cls, header: dict, eye: str = "left") -> "PicoCalibration":
        eye = _validate_eye(eye)
        raw_intrinsics = _numeric_tuple(header["cameraIntrinsics"], "cameraIntrinsics")
        if len(raw_intrinsics) != 4:
            raise ValueError("cameraIntrinsics must be [cx, cy, fx, fy]")
        pair = _matrix_pair(header["cameraExtrinsics"])
        if len(pair) <= PICO4_ULTRA_LEFT_EXTRINSIC_INDEX:
            raise ValueError("cameraExtrinsics does not contain header E1")
        if not all(np.isfinite(matrix).all() for matrix in pair):
            raise ValueError("cameraExtrinsics contains non-finite values")
        return cls(
            K=PICO4_ULTRA_KA.copy(),
            E_left=pair[PICO4_ULTRA_LEFT_EXTRINSIC_INDEX].copy(),
            E_pair=(pair[0].copy(), pair[1].copy()),
            D=PICO4_ULTRA_CAMERA_TO_IMAGE_D.copy(),
            raw_intrinsics=raw_intrinsics,
            left_extrinsic_index=PICO4_ULTRA_LEFT_EXTRINSIC_INDEX,
            eye=eye,
        )

    @classmethod
    def from_tracking_file(cls, path: str | Path, eye: str = "left") -> "PicoCalibration":
        return cls.from_header(_load_header(Path(path)), eye=eye)

    def as_dict(self) -> dict:
        return {
            "profile": self.camera_profile,
            "eye": self.eye,
            "K": self.K.tolist(),
            "E_eye": self.E_eye.tolist(),
            "E_left": self.E_left.tolist(),
            "E_right": self.E_right.tolist(),
            "E_header_index": self.extrinsic_index,
            "E_pair": [matrix.tolist() for matrix in self.E_pair],
            "D_camera_to_image": self.D.tolist(),
            "raw_intrinsics_cx_cy_fx_fy": list(self.raw_intrinsics),
            "formula": f"image_camera = D @ inv(H @ E_{self.eye}) @ hand_point",
            "hand_frame": "app_tracking_world",
            "head_pose": "direct",
        }
