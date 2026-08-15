"""対話式固定点ティーチングの変換部分を検証する。"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arm"))
sys.path.insert(0, str(ROOT / "arm" / "ik"))

import teach_fixed_poses as teaching  # noqa: E402
from solve_fixed_poses import rpy_matrix  # noqa: E402


def calibrations(direction=1):
    return [
        {
            "id": servo_id,
            "zero_tick": 2048,
            "direction": direction,
            "min_tick": 500,
            "max_tick": 3500,
        }
        for servo_id in range(1, 6)
    ]


def test_ticks_convert_to_joint_degrees():
    ticks = {servo_id: 2048 + servo_id * 10 for servo_id in range(1, 6)}
    result = teaching.joint_degrees_from_ticks(ticks, calibrations())
    expected = np.asarray([servo_id * 10 * 360 / 4096 for servo_id in range(1, 6)])
    assert np.allclose(result, expected)


def test_ticks_respect_direction():
    ticks = {servo_id: 2162 for servo_id in range(1, 6)}
    result = teaching.joint_degrees_from_ticks(ticks, calibrations(-1))
    assert np.allclose(result, -10.01953125)


def test_ticks_outside_safe_range_are_rejected():
    ticks = {servo_id: 2048 for servo_id in range(1, 6)}
    ticks[3] = 4000
    with pytest.raises(teaching.ConfigError, match="安全範囲"):
        teaching.joint_degrees_from_ticks(ticks, calibrations())


@pytest.mark.parametrize(
    "rpy",
    (
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([-80.0, 45.0, 175.0]),
        np.asarray([25.0, 90.0, 0.0]),
        np.asarray([-25.0, -90.0, 0.0]),
    ),
)
def test_rotation_matrix_roundtrip(rpy):
    rotation = rpy_matrix(rpy)
    recovered = teaching.matrix_to_rpy_degrees(rotation)
    assert np.allclose(rpy_matrix(recovered), rotation, atol=1e-9)


def test_write_poses_keeps_safety_lock_and_taught_seed(tmp_path):
    path = tmp_path / "poses.json"
    config = {"configured": True, "poses": {}}
    captures = {
        guide.name: {
            "joint_deg": np.asarray([1, 2, 3, 4, 5], dtype=float),
            "xyz_m": np.asarray([0.1, 0.2, 0.3], dtype=float),
            "rpy_deg": np.asarray([10, 20, 30], dtype=float),
        }
        for guide in teaching.POSE_GUIDES
    }
    teaching.write_poses(path, config, captures)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["configured"] is False
    assert saved["poses"]["HOME"] == {"joint_deg": [1.0, 2.0, 3.0, 4.0, 5.0]}
    assert saved["poses"]["PICK_FORK"]["seed_joint_deg"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_partial_teaching_preserves_other_poses(tmp_path):
    path = tmp_path / "poses.json"
    config = {
        "configured": False,
        "poses": {"PICK_FORK": {"xyz_m": [0.1, 0.2, 0.3], "seed": "HOME"}},
    }
    captures = {
        "HOME": {
            "joint_deg": np.asarray([1, 2, 3, 4, 5], dtype=float),
            "xyz_m": np.zeros(3),
            "rpy_deg": np.zeros(3),
        }
    }
    teaching.write_poses(path, config, captures)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["poses"]["PICK_FORK"] == config["poses"]["PICK_FORK"]
