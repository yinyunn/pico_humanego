"""读取 PICO trackingData JSONL，并保持设备原始坐标语义。"""

from __future__ import annotations

import json
from pathlib import Path


def _pose7(value: str | list[float]) -> dict:
    values = [float(item) for item in value.split(",")] if isinstance(value, str) else [float(item) for item in value]
    if len(values) != 7:
        raise ValueError("PICO pose must contain x,y,z,qx,qy,qz,qw")
    return {"position": values[:3], "quaternion_xyzw": values[3:], "pose7": values}


def _hand(raw: dict | None, side: str) -> dict | None:
    if not raw:
        return None
    joints = []
    for index, joint in enumerate(raw.get("HandJointLocations", [])):
        pose = _pose7(joint["p"])
        joints.append({
            "index": index,
            "name": joint.get("name"),
            "position": pose["position"],
            "quaternion_xyzw": pose["quaternion_xyzw"],
            "status": int(joint.get("s", 0)),
            "radius": float(joint.get("r", 0.0)),
        })
    if len(joints) != 26:
        raise ValueError(f"{side} hand must contain 26 joints, got {len(joints)}")
    return {
        "side": side,
        "is_active": int(raw.get("isActive", 0)),
        "scale": float(raw.get("scale", 1.0)),
        "joints": joints,
    }


def load_tracking(path: str | Path) -> tuple[dict, list[dict]]:
    """返回 ``(header, rows)``；手部位置不乘 Head.pose。"""
    header = None
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if header is None and "notice" in raw:
            header = raw
            continue
        if "Head" not in raw or "Hand" not in raw:
            continue
        head = _pose7(raw["Head"]["pose"])
        rows.append({
            "source_index": len(rows),
            "timestamp_ns": int(raw["timeStampNs"]),
            "predict_time_us": raw.get("predictTime"),
            "head": head,
            "head_status": raw["Head"].get("status"),
            "hands": {
                "left": _hand(raw["Hand"].get("leftHand"), "left"),
                "right": _hand(raw["Hand"].get("rightHand"), "right"),
            },
        })
    if header is None:
        raise ValueError(f"PICO tracking header not found: {path}")
    timestamps = [row["timestamp_ns"] for row in rows]
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("PICO tracking timestamps must be strictly increasing")
    return header, rows
