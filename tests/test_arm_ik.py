"""固定姿勢IKツールの、機材・PlaCo不要部分を検証する。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arm" / "ik"))

import solve_fixed_poses as ik  # noqa: E402


def calibration(direction=1):
    return ik.JointCalibration(1, 2048, direction, 0, 4095)


def test_degrees_to_ticks_uses_zero_and_direction():
    assert ik.degrees_to_ticks(np.array([0.0]), [calibration()]) == [2048]
    assert ik.degrees_to_ticks(np.array([90.0]), [calibration()]) == [3072]
    assert ik.degrees_to_ticks(np.array([90.0]), [calibration(-1)]) == [1024]


def test_degrees_to_ticks_rejects_joint_limit_violation():
    narrow = ik.JointCalibration(1, 2048, 1, 1900, 2200)
    with pytest.raises(ik.ConfigError, match="可動範囲"):
        ik.degrees_to_ticks(np.array([90.0]), [narrow])


def test_rpy_zero_is_identity():
    assert np.allclose(ik.rpy_matrix(np.zeros(3)), np.eye(3))


def test_rendered_header_is_explicitly_configured():
    cfg = {
        "joint_calibration": [
            {"id": i, "zero_tick": 2048, "direction": 1, "min_tick": 0, "max_tick": 4095}
            for i in range(1, 6)
        ],
        "gripper": {"id": 6, "open_tick": 2500, "closed_tick": 1800,
                    "min_tick": 1500, "max_tick": 2700},
    }
    solutions = {name: np.zeros(5) for name in ik.POSE_NAMES}
    header = ik.render_header(cfg, solutions)
    assert "POSE_TABLE_CONFIGURED = true" in header
    assert "GRIPPER_OPEN_TICK = 2500" in header
    assert "POSE_PICK_FORK = {{2048, 2048, 2048, 2048, 2048}}" in header


def test_taught_joint_seed_is_used_for_inverse_kinematics():
    class FakeKinematics:
        def __init__(self):
            self.seeds = []

        def inverse(self, seed_deg, desired, orientation_weight):
            self.seeds.append(np.asarray(seed_deg))
            return np.asarray(seed_deg)

        def forward(self, joint_deg):
            return np.eye(4)

    taught_seed = [1, 2, 3, 4, 5]
    poses = {"HOME": {"joint_deg": [0, 0, 0, 0, 0]}}
    for name in ik.POSE_NAMES[1:]:
        poses[name] = {
            "xyz_m": [0, 0, 0],
            "rpy_deg": [0, 0, 0],
            "seed": "HOME",
            "seed_joint_deg": taught_seed,
        }
    fake = FakeKinematics()
    solutions, errors = ik.solve_poses(
        {"poses": poses, "orientation_weight": 0.01}, kinematics=fake
    )
    assert all(np.allclose(seed, taught_seed) for seed in fake.seeds)
    assert all(error == pytest.approx(0) for error in errors.values())
    assert set(solutions) == set(ik.POSE_NAMES)


def test_trajectory_step_limit_checks_real_motion_paths():
    solutions = {name: np.zeros(5) for name in ik.POSE_NAMES}
    solutions["HOME"] = np.asarray([0, 0, 0, -150, 0], dtype=float)
    violations = ik.trajectory_step_violations(solutions, 120)
    assert ("fork", "HOME", "RETREAT", 150.0) in violations
    assert ("chopsticks", "RETREAT", "HOME", 150.0) in violations
