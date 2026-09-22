import json

import numpy as np

from tools.pico_import.calibration import PicoCalibration
from tools.pico_import.tracking_reader import load_tracking
from tools.pico_import.video_reader import crop_side_by_side, load_video_pts


def test_import_calibration_selects_header_e1_and_current_projection_contract():
    e0 = np.eye(4).tolist()
    e1 = np.eye(4).tolist()
    e1[0][3] = 0.1
    calibration = PicoCalibration.from_header({
        "cameraIntrinsics": "[1079.5,404.5,1373.6681390726,686.8680648828]",
        "cameraExtrinsics": f"{json.dumps(e0)}|{json.dumps(e1)}",
    })
    assert calibration.left_extrinsic_index == 1
    assert np.isclose(calibration.E_left[0, 3], 0.1)
    assert np.allclose(calibration.D, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.isclose(calibration.K[0, 0], 686.8340695363)


def test_import_calibration_selects_header_e0_for_right_eye():
    e0 = np.eye(4).tolist()
    e0[0][3] = -0.1
    e1 = np.eye(4).tolist()
    e1[0][3] = 0.1
    calibration = PicoCalibration.from_header({
        "cameraIntrinsics": "[1079.5,404.5,1373.6681390726,686.8680648828]",
        "cameraExtrinsics": f"{json.dumps(e0)}|{json.dumps(e1)}",
    }, eye="right")
    assert calibration.eye == "right"
    assert calibration.extrinsic_index == 0
    assert np.isclose(calibration.E_eye[0, 3], -0.1)
    assert calibration.camera_profile.endswith("right_e0_head_direct_handworld_axes_d")


def test_side_by_side_crop_selects_requested_eye():
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, :2] = 11
    frame[:, 2:] = 22
    assert np.all(crop_side_by_side(frame, "left", eye_width=2) == 11)
    assert np.all(crop_side_by_side(frame, "right", eye_width=2) == 22)


def test_tracking_reader_keeps_hand_points_in_raw_app_frame(tmp_path):
    joints = [{"p": "1,2,3,0,0,0,1", "s": 15, "r": 0.01} for _ in range(26)]
    row = {
        "timeStampNs": 100,
        "Head": {"pose": "0,0,0,0,0,0,1", "status": 3},
        "Hand": {
            "leftHand": {"isActive": 1, "count": 26, "HandJointLocations": joints},
            "rightHand": {"isActive": 1, "count": 26, "HandJointLocations": joints},
        },
    }
    path = tmp_path / "trackingData.txt"
    path.write_text(
        json.dumps({"notice": "header", "cameraIntrinsics": "[1,2,3,4]", "cameraExtrinsics": "[]"})
        + "\n"
        + json.dumps(row)
        + "\n",
        encoding="utf-8",
    )
    _, records = load_tracking(path)
    assert records[0]["hands"]["left"]["joints"][0]["position"] == [1.0, 2.0, 3.0]
    assert records[0]["head"]["position"] == [0.0, 0.0, 0.0]


def test_video_pts_reader_rejects_non_monotonic_frames(tmp_path):
    path = tmp_path / "pts.jsonl"
    path.write_text(
        json.dumps({"frame_index": 1, "pts_seconds": 0.1})
        + "\n"
        + json.dumps({"frame_index": 0, "pts_seconds": 0.2})
        + "\n",
        encoding="utf-8",
    )
    try:
        load_video_pts(path)
    except ValueError as exc:
        assert "frame indices" in str(exc)
    else:
        raise AssertionError("non-monotonic PTS frame indices were accepted")


def test_stereo_2d_keeps_eye_pixels_and_does_not_export_depth():
    from types import SimpleNamespace
    from scripts.visualize_mediapipe_mp4 import _detect_eye

    class Detector:
        def detect_for_video(self, image, timestamp):
            assert timestamp == 123
            # OpenCV BGR is converted to RGB before inference.
            assert image.numpy_view()[0, 0].tolist() == [30, 20, 10]
            return SimpleNamespace(
                hand_landmarks=[[SimpleNamespace(x=0.255, y=0.375, z=-99) for _ in range(21)]],
                handedness=[[SimpleNamespace(category_name="Left", score=0.8)]],
            )

    image = np.full((80, 100, 3), [10, 20, 30], dtype=np.uint8)
    hand = _detect_eye(image, Detector(), 123, "right", 100, False)[0]
    assert hand["joints"][0]["u_eye"] == 25.5
    assert hand["joints"][0]["u"] == 125.5
    assert hand["joints"][0]["v_eye"] == 30.0
    assert "z_norm" not in hand["joints"][0]
    assert len(hand["joints"]) == 21


def test_stereo_candidates_are_not_verified_and_flag_ambiguity():
    from scripts.visualize_mediapipe_mp4 import _match_stereo_hands

    left = [{"hand_side": "Left", "score": .8, "eye_hand_id": 0},
            {"hand_side": "Left", "score": .7, "eye_hand_id": 1}]
    right = [{"hand_side": "Left", "score": .9, "eye_hand_id": 1}]
    matched, missing = _match_stereo_hands(left, right)
    assert matched["matched"] and matched["ambiguous"]
    assert not matched["geometry_verified"]
    assert not missing["matched"]


def test_stereo_smoke_does_not_write_final_hand_labels(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.pico_import.humanego_writer import HumanEgoWriter
    from scripts import visualize_mediapipe_mp4

    calls = []
    session = SimpleNamespace(video_path=tmp_path / "input.mp4", pts_path=tmp_path / "pts.jsonl",
                              write_intermediate=lambda output, **kwargs: calls.append(output))
    monkeypatch.setattr(visualize_mediapipe_mp4, "main", lambda argv: {"argv": argv})
    result = HumanEgoWriter(session, tmp_path).export_stereo_2d_smoke(max_frames=2)
    assert calls == [tmp_path / "stereo_2d"]
    assert "--pts" in result["argv"] and "--sync-manifest" in result["argv"]
    assert result["argv"][-2:] == ["--max-frames", "2"]
    assert not (tmp_path / "preprocess").exists()
