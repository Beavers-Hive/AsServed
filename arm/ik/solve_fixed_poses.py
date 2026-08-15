"""SO-101の固定作業点をPlaCo IKで解き、XIAO用の姿勢テーブルを生成する。

本番中にIKは解かない。ラックと提供先を実測して fixed_poses.json に入力し、
このツールで一度だけ関節角→STS3215 raw tickへ変換する。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


POSE_NAMES = (
    "HOME",
    "APPROACH_FORK",
    "PICK_FORK",
    "APPROACH_CHOPSTICKS",
    "PICK_CHOPSTICKS",
    "APPROACH_PLACE",
    "PLACE_UTENSIL",
    "RETREAT",
)
EXPECTED_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
TICKS_PER_DEGREE = 4096.0 / 360.0
MOTION_PATHS = {
    "fork": (
        "HOME", "RETREAT", "APPROACH_FORK", "PICK_FORK", "APPROACH_FORK",
        "APPROACH_PLACE", "PLACE_UTENSIL", "APPROACH_PLACE", "RETREAT", "HOME",
    ),
    "chopsticks": (
        "HOME", "RETREAT", "APPROACH_CHOPSTICKS", "PICK_CHOPSTICKS",
        "APPROACH_CHOPSTICKS", "APPROACH_PLACE", "PLACE_UTENSIL",
        "APPROACH_PLACE", "RETREAT", "HOME",
    ),
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class JointCalibration:
    servo_id: int
    zero_tick: int
    direction: int
    min_tick: int
    max_tick: int


def _numbers(value: Any, n: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != n or any(v is None for v in value):
        raise ConfigError(f"{label} は {n} 個の数値を指定してください")
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} に数値以外が含まれています") from exc


def load_config(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw.get("configured", False):
        raise ConfigError(
            "configured=false です。実測値・校正値を入力し、安全確認後に true にしてください"
        )
    if tuple(raw.get("joint_order", ())) != EXPECTED_JOINTS:
        raise ConfigError(f"joint_order は {list(EXPECTED_JOINTS)} の順にしてください")

    calibrations = raw.get("joint_calibration")
    if not isinstance(calibrations, list) or len(calibrations) != 5:
        raise ConfigError("joint_calibration は本体5関節分必要です")
    for i, item in enumerate(calibrations, 1):
        for key in ("id", "zero_tick", "direction", "min_tick", "max_tick"):
            if item.get(key) is None:
                raise ConfigError(f"joint_calibration[{i - 1}].{key} が未入力です")
        if int(item["id"]) != i:
            raise ConfigError(f"joint_calibration[{i - 1}].id は {i} にしてください")
        if int(item["direction"]) not in (-1, 1):
            raise ConfigError(f"joint_calibration[{i - 1}].direction は -1 または 1 です")
        if not 0 <= int(item["min_tick"]) < int(item["max_tick"]) <= 4095:
            raise ConfigError(f"joint_calibration[{i - 1}] のmin/max tickが不正です")

    gripper = raw.get("gripper", {})
    for key in ("id", "open_tick", "closed_tick", "min_tick", "max_tick"):
        if gripper.get(key) is None:
            raise ConfigError(f"gripper.{key} が未入力です")
    if int(gripper["id"]) != 6:
        raise ConfigError("gripper.id は 6 にしてください")
    if not 0 <= int(gripper["min_tick"]) < int(gripper["max_tick"]) <= 4095:
        raise ConfigError("gripper のmin/max tickが不正です")
    for key in ("open_tick", "closed_tick"):
        if not int(gripper["min_tick"]) <= int(gripper[key]) <= int(gripper["max_tick"]):
            raise ConfigError(f"gripper.{key} が可動範囲外です")

    poses = raw.get("poses", {})
    for name in POSE_NAMES:
        if name not in poses:
            raise ConfigError(f"poses.{name} がありません")
        pose = poses[name]
        if "joint_deg" in pose:
            _numbers(pose["joint_deg"], 5, f"poses.{name}.joint_deg")
        else:
            _numbers(pose.get("xyz_m"), 3, f"poses.{name}.xyz_m")
            _numbers(pose.get("rpy_deg"), 3, f"poses.{name}.rpy_deg")
            seed = pose.get("seed")
            if seed not in POSE_NAMES:
                raise ConfigError(f"poses.{name}.seed が不正です: {seed!r}")
            if "seed_joint_deg" in pose:
                _numbers(pose["seed_joint_deg"], 5, f"poses.{name}.seed_joint_deg")

    urdf = Path(raw.get("urdf", ""))
    if not urdf.is_absolute():
        urdf = Path.cwd() / urdf
    if not urdf.exists():
        raise ConfigError(f"URDFがありません: {urdf}")
    raw["_urdf_path"] = urdf.resolve()
    return raw


def calibrations_from_config(cfg: dict) -> list[JointCalibration]:
    return [
        JointCalibration(
            servo_id=int(item["id"]),
            zero_tick=int(item["zero_tick"]),
            direction=int(item["direction"]),
            min_tick=int(item["min_tick"]),
            max_tick=int(item["max_tick"]),
        )
        for item in cfg["joint_calibration"]
    ]


def degrees_to_ticks(joint_deg: np.ndarray, calibrations: list[JointCalibration]) -> list[int]:
    if len(joint_deg) != len(calibrations):
        raise ConfigError("関節角と校正値の個数が一致しません")
    ticks: list[int] = []
    for deg, cal in zip(joint_deg, calibrations):
        raw = int(round(cal.zero_tick + cal.direction * float(deg) * TICKS_PER_DEGREE))
        if not cal.min_tick <= raw <= cal.max_tick:
            raise ConfigError(
                f"servo {cal.servo_id}: {deg:.2f}deg → {raw}tick は "
                f"可動範囲 {cal.min_tick}..{cal.max_tick} 外です"
            )
        ticks.append(raw)
    return ticks


def rpy_matrix(rpy_deg: np.ndarray) -> np.ndarray:
    """URDFと同じ固定軸RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    roll, pitch, yaw = np.deg2rad(rpy_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


class PlacoKinematics:
    """LeRobotのRobotKinematicsと同じPlaCo呼び出しを固定姿勢生成用に薄く包む。"""

    def __init__(self, urdf_path: Path, target_frame: str, joint_names: list[str]):
        try:
            import placo
        except ImportError as exc:
            raise ConfigError(
                "placo がありません。arm/ik の専用venvへ requirements.txt をインストールしてください"
            ) from exc
        # このツールは固定姿勢のFK/IKだけを行う。URDFの隣接リンク同士を含む
        # collision mesh警告は使わず、実機の干渉は各ウェイポイントで確認する。
        self.robot = placo.RobotWrapper(str(urdf_path), placo.Flags.ignore_collisions)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.target_frame = target_frame
        self.joint_names = joint_names
        self.task = self.solver.add_frame_task(target_frame, np.eye(4))

    def _set_joints(self, joint_deg: np.ndarray) -> None:
        for name, value in zip(self.joint_names, np.deg2rad(joint_deg)):
            self.robot.set_joint(name, float(value))

    def forward(self, joint_deg: np.ndarray) -> np.ndarray:
        self._set_joints(joint_deg)
        self.robot.update_kinematics()
        return np.asarray(self.robot.get_T_world_frame(self.target_frame), dtype=float)

    def inverse(self, seed_deg: np.ndarray, desired: np.ndarray,
                orientation_weight: float) -> np.ndarray:
        self._set_joints(seed_deg)
        self.task.T_world_frame = desired
        self.task.configure(self.target_frame, "soft", 1.0, orientation_weight)
        # LeRobotのリアルタイム処理は制御ループごとに1反復する。ここではオフライン生成
        # なので、同じ処理を収束まで反復する。1回だけだとseed近傍の微小更新しか得られない。
        for _ in range(100):
            self.robot.update_kinematics()
            self.solver.solve(True)
            self.robot.update_kinematics()
            actual = np.asarray(self.robot.get_T_world_frame(self.target_frame), dtype=float)
            if np.linalg.norm(actual[:3, 3] - desired[:3, 3]) < 1e-6:
                break
        return np.rad2deg([self.robot.get_joint(name) for name in self.joint_names])


def solve_poses(cfg: dict, kinematics=None) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    kin = kinematics or PlacoKinematics(
        cfg["_urdf_path"], cfg.get("target_frame", "gripper_frame_link"),
        list(cfg["joint_order"]),
    )
    poses = cfg["poses"]
    solutions: dict[str, np.ndarray] = {}
    errors_mm: dict[str, float] = {}
    orientation_weight = float(cfg.get("orientation_weight", 0.01))

    # 明示関節角を先に登録し、IK姿勢がHOMEなどをseedとして参照できるようにする。
    for name in POSE_NAMES:
        if "joint_deg" in poses[name]:
            solutions[name] = _numbers(poses[name]["joint_deg"], 5, f"poses.{name}.joint_deg")
            errors_mm[name] = 0.0

    pending = [name for name in POSE_NAMES if name not in solutions]
    while pending:
        progressed = False
        for name in pending[:]:
            pose = poses[name]
            seed_name = pose["seed"]
            if seed_name not in solutions:
                continue
            target = np.eye(4, dtype=float)
            target[:3, :3] = rpy_matrix(_numbers(pose["rpy_deg"], 3, f"poses.{name}.rpy_deg"))
            target[:3, 3] = _numbers(pose["xyz_m"], 3, f"poses.{name}.xyz_m")
            seed_deg = (
                _numbers(pose["seed_joint_deg"], 5, f"poses.{name}.seed_joint_deg")
                if "seed_joint_deg" in pose
                else solutions[seed_name]
            )
            result = np.asarray(
                kin.inverse(seed_deg, target, orientation_weight), dtype=float
            )
            actual = kin.forward(result)
            errors_mm[name] = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]) * 1000.0)
            solutions[name] = result
            pending.remove(name)
            progressed = True
        if not progressed:
            raise ConfigError(f"seed の循環または未解決参照があります: {pending}")
    return solutions, errors_mm


def trajectory_step_violations(
    solutions: dict[str, np.ndarray], max_step_deg: float
) -> list[tuple[str, str, str, float]]:
    """実際の再生順で、単一関節の変化が大きすぎる区間を返す。"""
    violations: list[tuple[str, str, str, float]] = []
    for path_name, sequence in MOTION_PATHS.items():
        for start, end in zip(sequence, sequence[1:]):
            maximum = float(np.max(np.abs(solutions[end] - solutions[start])))
            if maximum > max_step_deg:
                violations.append((path_name, start, end, maximum))
    return violations


def render_header(cfg: dict, solutions: dict[str, np.ndarray]) -> str:
    calibrations = calibrations_from_config(cfg)
    ticks = {name: degrees_to_ticks(solutions[name], calibrations) for name in POSE_NAMES}
    gripper = cfg["gripper"]
    mins = [c.min_tick for c in calibrations] + [int(gripper["min_tick"])]
    maxs = [c.max_tick for c in calibrations] + [int(gripper["max_tick"])]
    ids = [c.servo_id for c in calibrations] + [int(gripper["id"])]

    def values(items) -> str:
        return ", ".join(str(int(x)) for x in items)

    lines = [
        "#pragma once",
        "",
        "#include <Arduino.h>",
        "",
        "// Generated by arm/ik/solve_fixed_poses.py. Do not edit by hand.",
        "static constexpr bool POSE_TABLE_CONFIGURED = true;",
        "",
        "struct ArmPose { int16_t body[5]; };",
        "",
        f"static constexpr uint8_t SERVO_IDS[6] = {{{values(ids)}}};",
        f"static constexpr int16_t JOINT_MIN_TICKS[6] = {{{values(mins)}}};",
        f"static constexpr int16_t JOINT_MAX_TICKS[6] = {{{values(maxs)}}};",
        f"static constexpr int16_t GRIPPER_OPEN_TICK = {int(gripper['open_tick'])};",
        f"static constexpr int16_t GRIPPER_CLOSED_TICK = {int(gripper['closed_tick'])};",
        "",
    ]
    for name in POSE_NAMES:
        lines.append(f"static constexpr ArmPose POSE_{name} = {{{{{values(ticks[name])}}}}};")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="SO-101固定点IK → XIAO姿勢テーブル")
    parser.add_argument("--config", type=Path, default=here / "fixed_poses.json")
    parser.add_argument(
        "--output", type=Path,
        default=here.parent / "firmware_xiao" / "include" / "generated_poses.h",
    )
    parser.add_argument("--check", action="store_true", help="検証だけ行いheaderを書かない")
    parser.add_argument("--max-error-mm", type=float, default=3.0)
    parser.add_argument(
        "--max-joint-step-deg", type=float, default=120.0,
        help="1区間の単一関節に許す最大変化角（既定: 120度）",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        solutions, errors = solve_poses(cfg)
        # IK姿勢は位置誤差を必ず検証する。HOMEの明示関節角は0扱い。
        bad = {name: error for name, error in errors.items() if error > args.max_error_mm}
        for name in POSE_NAMES:
            q = ", ".join(f"{x:7.2f}" for x in solutions[name])
            print(f"[ik] {name:<22} [{q}] deg  error={errors[name]:.2f}mm")
        if bad:
            detail = ", ".join(f"{k}={v:.2f}mm" for k, v in bad.items())
            raise ConfigError(f"IK位置誤差が {args.max_error_mm:g}mm を超えています: {detail}")
        large_steps = trajectory_step_violations(solutions, args.max_joint_step_deg)
        if large_steps:
            detail = ", ".join(
                f"{path}:{start}→{end}={step:.1f}deg"
                for path, start, end, step in large_steps
            )
            raise ConfigError(
                f"1区間の関節変化が {args.max_joint_step_deg:g}度を超えています: {detail}"
            )
        header = render_header(cfg, solutions)
        if args.check:
            print("[ik] 検証のみ: headerは変更していません")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(header, encoding="utf-8")
            print(f"[ik] XIAO姿勢テーブルを書き出しました: {args.output}")
        return 0
    except (ConfigError, json.JSONDecodeError, OSError) as exc:
        print(f"[ik] エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
