"""対話式SO-101校正の計算部分を検証する。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "arm"))

import calibrate_xiao as calibration  # noqa: E402


def test_wrapped_delta_handles_12_bit_wrap():
    assert calibration.wrapped_delta(100, 4000) == 196
    assert calibration.wrapped_delta(3900, 100) == -296


def test_joint_direction_increasing():
    result = calibration.calculate_joint(1, 2048, 900, 3200, 80)
    assert result == {
        "id": 1,
        "zero_tick": 2048,
        "direction": 1,
        "min_tick": 980,
        "max_tick": 3120,
    }


def test_joint_direction_decreasing():
    result = calibration.calculate_joint(2, 2048, 3200, 900, 80)
    assert result["direction"] == -1
    assert result["min_tick"] == 980
    assert result["max_tick"] == 3120


def test_joint_rejects_wrapped_numeric_range():
    with pytest.raises(calibration.WraparoundError, match="0/4095"):
        calibration.calculate_joint(1, 20, 3900, 300, 20)


def test_joint_rejects_center_outside_endpoints():
    with pytest.raises(calibration.CalibrationError, match="中央"):
        calibration.calculate_joint(1, 1000, 1300, 1700, 20)


def test_gripper_range_adds_padding():
    result = calibration.calculate_gripper(1800, 2300, 20)
    assert result == {
        "id": 6,
        "open_tick": 1800,
        "closed_tick": 2300,
        "min_tick": 1780,
        "max_tick": 2320,
    }
