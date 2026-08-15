"""SO-101を手で動かして固定作業点を対話的にティーチングする。

校正済みraw tickを関節角へ変換し、公式URDFの順運動学でTCPのXYZ/RPYを求める。
記録中はトルクOFFで、XIAOへ移動指令は送らない。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import serial

from calibrate_xiao import CalibrationError, XiaoConnection, wrapped_delta

import sys

IK_DIR = Path(__file__).resolve().parent / "ik"
sys.path.insert(0, str(IK_DIR))
from solve_fixed_poses import (  # noqa: E402
    EXPECTED_JOINTS,
    PlacoKinematics,
    ConfigError,
)


TICKS_PER_DEGREE = 4096.0 / 360.0


@dataclass(frozen=True)
class PoseGuide:
    name: str
    title: str
    instruction: str
    seed: str | None


POSE_GUIDES = (
    PoseGuide(
        "HOME",
        "待機姿勢",
        "ラック、机、投影範囲から離れた安全な待機姿勢へ動かします。",
        None,
    ),
    PoseGuide(
        "APPROACH_FORK",
        "フォーク上方",
        "開いたグリッパーをフォークの持ち手の約50〜60mm上へ合わせます。",
        "HOME",
    ),
    PoseGuide(
        "PICK_FORK",
        "フォーク把持位置",
        "開いた爪の間へフォークの持ち手が入る位置まで下げます。まだ挟みません。",
        "APPROACH_FORK",
    ),
    PoseGuide(
        "APPROACH_CHOPSTICKS",
        "箸上方",
        "開いたグリッパーを箸の把持点の約50〜60mm上へ合わせます。",
        "HOME",
    ),
    PoseGuide(
        "PICK_CHOPSTICKS",
        "箸把持位置",
        "開いた爪の間へ箸の把持部分が入る位置まで下げます。まだ挟みません。",
        "APPROACH_CHOPSTICKS",
    ),
    PoseGuide(
        "APPROACH_PLACE",
        "提供位置上方",
        "食器を置く位置の約50〜60mm上へ、グリッパーを安全に合わせます。",
        "HOME",
    ),
    PoseGuide(
        "PLACE_UTENSIL",
        "提供位置",
        "食器を離したい高さまで下げます。机へ爪を押し付けないでください。",
        "APPROACH_PLACE",
    ),
    PoseGuide(
        "RETREAT",
        "中央通過・退避姿勢",
        "机やラックより高い中央位置で、HOMEと各作業点の間を安全に通過できる姿勢へ動かします。",
        "APPROACH_PLACE",
    ),
)


def joint_degrees_from_ticks(
    ticks: dict[int, int], calibrations: list[dict]
) -> np.ndarray:
    if len(calibrations) != 5:
        raise ConfigError("先に手順3を完了してください: 本体5軸の校正値がありません")
    degrees: list[float] = []
    for expected_id, item in enumerate(calibrations, 1):
        for key in ("id", "zero_tick", "direction", "min_tick", "max_tick"):
            if item.get(key) is None:
                raise ConfigError(f"先に手順3を完了してください: ID {expected_id} の{key}が未入力です")
        servo_id = int(item["id"])
        if servo_id != expected_id or servo_id not in ticks:
            raise ConfigError(f"校正IDが不正です: {servo_id}")
        tick = int(ticks[servo_id])
        minimum, maximum = int(item["min_tick"]), int(item["max_tick"])
        if not minimum <= tick <= maximum:
            raise ConfigError(
                f"ID {servo_id}: {tick}tick は安全範囲 {minimum}..{maximum} 外です"
            )
        direction = int(item["direction"])
        if direction not in (-1, 1):
            raise ConfigError(f"ID {servo_id}: directionは-1または1にしてください")
        delta = wrapped_delta(tick, int(item["zero_tick"]))
        degrees.append(direction * delta / TICKS_PER_DEGREE)
    return np.asarray(degrees, dtype=float)


def matrix_to_rpy_degrees(rotation: np.ndarray) -> np.ndarray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll)の回転行列を固定軸RPYへ戻す。"""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotationは3x3で指定してください")
    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.hypot(float(rotation[0, 0]), float(rotation[1, 0])),
    )
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    elif pitch > 0:
        # 特異点ではyaw=0を選び、等価なrollへまとめる。
        yaw = 0.0
        roll = math.atan2(float(rotation[0, 1]), float(rotation[0, 2]))
    else:
        yaw = 0.0
        roll = math.atan2(-float(rotation[0, 1]), -float(rotation[0, 2]))
    return np.rad2deg([roll, pitch, yaw])


def recorded_pose(
    ticks: dict[int, int], calibrations: list[dict], kinematics: PlacoKinematics
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_deg = joint_degrees_from_ticks(ticks, calibrations)
    transform = kinematics.forward(joint_deg)
    xyz_m = np.asarray(transform[:3, 3], dtype=float)
    rpy_deg = matrix_to_rpy_degrees(transform[:3, :3])
    return joint_deg, xyz_m, rpy_deg


def rounded(values: np.ndarray, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


def write_poses(path: Path, config: dict, captures: dict[str, dict]) -> None:
    poses: dict[str, dict] = dict(config.get("poses", {}))
    for guide in POSE_GUIDES:
        if guide.name not in captures:
            continue
        captured = captures[guide.name]
        if guide.name == "HOME":
            poses[guide.name] = {"joint_deg": rounded(captured["joint_deg"], 3)}
        else:
            poses[guide.name] = {
                "xyz_m": rounded(captured["xyz_m"], 6),
                "rpy_deg": rounded(captured["rpy_deg"], 3),
                "seed": guide.seed,
                # 実測姿勢のIK枝から解き始め、別解へのジャンプを防ぐ。
                "seed_joint_deg": rounded(captured["joint_deg"], 3),
            }
    config["poses"] = poses
    config["configured"] = False

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(config, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="SO-101固定作業点の対話式ティーチング")
    parser.add_argument("--port", required=True, help="例: /dev/cu.usbmodem1101")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--config", type=Path, default=here / "ik" / "fixed_poses.json")
    parser.add_argument(
        "--poses",
        nargs="+",
        choices=[guide.name for guide in POSE_GUIDES],
        help="指定した姿勢だけ再ティーチング。省略時は全8姿勢",
    )
    args = parser.parse_args()

    connection: XiaoConnection | None = None
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        calibrations = config.get("joint_calibration", [])
        urdf = Path(config.get("urdf", ""))
        if not urdf.is_absolute():
            urdf = Path.cwd() / urdf
        if not urdf.exists():
            raise ConfigError(f"URDFがありません: {urdf}")
        if tuple(config.get("joint_order", ())) != EXPECTED_JOINTS:
            raise ConfigError("joint_orderがSO-101の5軸順と一致しません")

        print("SO-101 固定作業点の対話式ティーチング")
        print("- 食器をつかまず、グリッパーを開いたまま位置だけ合わせます")
        print("- アームを手で支え、机やラックへ押し付けないでください")
        print("- 移動命令は送らず、接続中はトルクOFFです")
        answer = input("\n準備できたら Enter（中止は q + Enter）: ").strip().lower()
        if answer == "q":
            raise KeyboardInterrupt

        connection = XiaoConnection(args.port, args.baud)
        connection.torque_off()
        kinematics = PlacoKinematics(
            urdf.resolve(),
            config.get("target_frame", "gripper_frame_link"),
            list(config["joint_order"]),
        )

        selected_names = set(args.poses or [guide.name for guide in POSE_GUIDES])
        selected_guides = [guide for guide in POSE_GUIDES if guide.name in selected_names]
        captures: dict[str, dict] = {}
        for index, guide in enumerate(selected_guides, 1):
            while True:
                print(f"\n--- {index}/{len(selected_guides)} {guide.name}: {guide.title} ---")
                print(guide.instruction)
                answer = input("位置を合わせたら Enter（qで中止）: ").strip().lower()
                if answer == "q":
                    raise KeyboardInterrupt
                ticks = connection.read_joints()
                joint_deg, xyz_m, rpy_deg = recorded_pose(ticks, calibrations, kinematics)
                print("ticks :", " ".join(f"{i}:{ticks[i]}" for i in range(1, 6)))
                print("joint :", " ".join(f"{value:+7.2f}°" for value in joint_deg))
                print("TCP   :", " ".join(f"{value * 1000:+7.1f}mm" for value in xyz_m))
                print("RPY   :", " ".join(f"{value:+7.1f}°" for value in rpy_deg))
                answer = input("Enterで採用、rで位置を合わせ直す、qで中止: ").strip().lower()
                if answer == "q":
                    raise KeyboardInterrupt
                if answer == "r":
                    continue
                captures[guide.name] = {
                    "ticks": ticks,
                    "joint_deg": joint_deg,
                    "xyz_m": xyz_m,
                    "rpy_deg": rpy_deg,
                }
                break

        print("\n=== 記録したTCP位置 ===")
        for guide in selected_guides:
            xyz_mm = captures[guide.name]["xyz_m"] * 1000.0
            print(
                f"{guide.name:<22} "
                f"x={xyz_mm[0]:+7.1f} y={xyz_mm[1]:+7.1f} z={xyz_mm[2]:+7.1f} mm"
            )
        answer = input(
            f"\nEnterで {args.config} へ保存（nで保存せず終了）: "
        ).strip().lower()
        if answer == "n":
            print("保存しませんでした。")
            return 0
        write_poses(args.config, config, captures)
        print(f"保存しました: {args.config}")
        print("configured=falseを維持しています。次はIK検証です。")
        return 0
    except KeyboardInterrupt:
        print("\nティーチングを中止しました。設定ファイルは変更していません。")
        return 130
    except (
        CalibrationError,
        ConfigError,
        OSError,
        serial.SerialException,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"ティーチングエラー: {exc}")
        return 1
    finally:
        if connection is not None:
            try:
                connection.torque_off()
            except (CalibrationError, OSError, serial.SerialException):
                pass
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
