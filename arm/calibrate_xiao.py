"""XIAO経由でSO-101の関節校正を対話的に行う。

このツールはサーボへ移動指令を送らない。接続中はトルクを解除し、利用者が
各関節を手で動かした位置を読み取って fixed_poses.json を更新する。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import serial


TICK_COUNT = 4096
HALF_TURN_TICKS = TICK_COUNT // 2
JOINTS_RE = re.compile(r"^JOINTS((?:\s+\d+:-?\d+)+)$")
VALUE_RE = re.compile(r"(\d+):(-?\d+)")


class CalibrationError(RuntimeError):
    pass


class WraparoundError(CalibrationError):
    """サーボの0/4095折り返しが関節可動域内にある。"""


@dataclass(frozen=True)
class JointGuide:
    servo_id: int
    name: str
    negative_hint: str
    positive_hint: str


JOINT_GUIDES = (
    JointGuide(1, "ベース回転", "上から見て反時計回り（先端をアームの左へ）",
               "上から見て時計回り（先端をアームの右へ）"),
    JointGuide(2, "肩", "前方へ伸ばした上腕を上げる方向",
               "前方へ伸ばした上腕を下げる方向"),
    JointGuide(3, "肘", "前腕を上へ曲げる方向", "前腕を下へ曲げる方向"),
    JointGuide(4, "手首上下", "グリッパー先端を上げる方向",
               "グリッパー先端を下げる方向"),
    JointGuide(5, "手首回転", "先端から本体を見て時計回り",
               "先端から本体を見て反時計回り"),
)


def wrapped_delta(value: int, center: int) -> int:
    """12-bit位置のcenterからvalueへの最短符号付き差分。"""
    return (value - center + HALF_TURN_TICKS) % TICK_COUNT - HALF_TURN_TICKS


def calculate_joint(
    servo_id: int,
    center: int,
    negative: int,
    positive: int,
    margin: int,
) -> dict[str, int]:
    """中央、URDF負側、URDF正側の実測値から校正値を作る。"""
    for label, value in (("中央", center), ("負側", negative), ("正側", positive)):
        if not 0 <= value < TICK_COUNT:
            raise CalibrationError(f"ID {servo_id} の{label}が範囲外です: {value}")

    d_negative = wrapped_delta(negative, center)
    d_positive = wrapped_delta(positive, center)
    if abs(d_negative) < 80 or abs(d_positive) < 80:
        raise CalibrationError(
            f"ID {servo_id}: 端点が中央に近すぎます。もっと大きく動かして再測定してください"
        )
    if d_negative * d_positive >= 0:
        raise CalibrationError(
            f"ID {servo_id}: 中央が2つの端点の間にありません。中央姿勢または端点を再測定してください"
        )

    # 現在のファームウェアの範囲判定は単純な min <= tick <= max。
    # したがって0/4095をまたぐ可動域は、誤指令防止のため保存しない。
    raw_min, raw_max = sorted((negative, positive))
    if not raw_min < center < raw_max:
        raise WraparoundError(
            f"ID {servo_id}: 0/4095をまたぐ範囲です。このままでは安全に扱えません"
        )

    safe_min = raw_min + margin
    safe_max = raw_max - margin
    if safe_min >= safe_max or not safe_min < center < safe_max:
        raise CalibrationError(
            f"ID {servo_id}: 安全余白{margin}tickを取れません。端点を再測定してください"
        )

    # 物理的な正方向でraw tickが増えるなら+1、減るなら-1。
    direction = 1 if d_positive > d_negative else -1
    return {
        "id": servo_id,
        "zero_tick": center,
        "direction": direction,
        "min_tick": safe_min,
        "max_tick": safe_max,
    }


def calculate_gripper(open_tick: int, closed_tick: int, padding: int) -> dict[str, int]:
    for label, value in (("開位置", open_tick), ("閉位置", closed_tick)):
        if not 0 <= value < TICK_COUNT:
            raise CalibrationError(f"グリッパーの{label}が範囲外です: {value}")
    if abs(open_tick - closed_tick) < 30:
        raise CalibrationError("グリッパーの開位置と閉位置が近すぎます")
    if abs(open_tick - closed_tick) > HALF_TURN_TICKS:
        raise CalibrationError("グリッパーが0/4095をまたいでいます")
    return {
        "id": 6,
        "open_tick": open_tick,
        "closed_tick": closed_tick,
        "min_tick": max(0, min(open_tick, closed_tick) - padding),
        "max_tick": min(TICK_COUNT - 1, max(open_tick, closed_tick) + padding),
    }


class XiaoConnection:
    def __init__(self, port: str, baud: int):
        self.serial = serial.Serial(port, baud, timeout=0.1)
        # USB CDCを開くとXIAOが再起動するため、setup完了まで待つ。
        time.sleep(2.2)
        self.serial.reset_input_buffer()

    def close(self) -> None:
        self.serial.close()

    def command(self, command: str) -> None:
        self.serial.write((command + "\n").encode("ascii"))
        self.serial.flush()

    def torque_off(self) -> None:
        self.command("TORQUE_OFF")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            line = self.serial.readline().decode("ascii", "replace").strip()
            if line == "OK TORQUE_OFF":
                return
        raise CalibrationError("XIAOからTORQUE_OFFの応答がありません")

    def read_joints(self) -> dict[int, int]:
        self.command("READ_JOINTS")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            line = self.serial.readline().decode("ascii", "replace").strip()
            match = JOINTS_RE.match(line)
            if not match:
                continue
            values = {int(i): int(value) for i, value in VALUE_RE.findall(match.group(1))}
            if set(values) != set(range(1, 7)):
                raise CalibrationError(f"6軸すべてを読めませんでした: {line}")
            offline = [servo_id for servo_id, value in values.items() if value < 0]
            if offline:
                raise CalibrationError(f"応答しないサーボがあります: ID {offline}")
            if any(value >= TICK_COUNT for value in values.values()):
                raise CalibrationError(f"不正なサーボ値です: {line}")
            return values
        raise CalibrationError("XIAOからREAD_JOINTSの応答がありません")

    def calibrate_center(self, servo_id: int) -> int:
        """現在の物理姿勢をSTS3215の中位として永続設定する。"""
        self.command(f"CALIBRATE_CENTER {servo_id} CONFIRM")
        deadline = time.monotonic() + 3.0
        prefix = f"OK CALIBRATE_CENTER id={servo_id} "
        while time.monotonic() < deadline:
            line = self.serial.readline().decode("ascii", "replace").strip()
            if line.startswith(prefix):
                match = re.search(r"after=(-?\d+)", line)
                if match is None:
                    raise CalibrationError(f"中位校正の応答を解釈できません: {line}")
                return int(match.group(1))
            if line.startswith("ERR "):
                raise CalibrationError(f"XIAOの中位校正に失敗しました: {line}")
        raise CalibrationError("XIAOから中位校正の応答がありません。ファームウェアを更新してください")


def wait_to_capture(message: str) -> None:
    answer = input(f"\n{message}\n準備できたら Enter（中止は q + Enter）: ").strip().lower()
    if answer == "q":
        raise KeyboardInterrupt


def capture(connection: XiaoConnection, servo_id: int, message: str) -> int:
    wait_to_capture(message)
    values = connection.read_joints()
    value = values[servo_id]
    print(f"  → ID {servo_id}: {value} tick")
    return value


def print_summary(joints: list[dict[str, int]], gripper: dict[str, int]) -> None:
    print("\n=== 計算結果 ===")
    print("ID  zero  dir   min   max")
    for item in joints:
        print(
            f"{item['id']:>2}  {item['zero_tick']:>4}  {item['direction']:>+3}  "
            f"{item['min_tick']:>4}  {item['max_tick']:>4}"
        )
    print(
        "gripper: "
        f"open={gripper['open_tick']} closed={gripper['closed_tick']} "
        f"min={gripper['min_tick']} max={gripper['max_tick']}"
    )


def save_config(path: Path, joints: list[dict[str, int]], gripper: dict[str, int]) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["joint_calibration"] = joints
    config["gripper"] = gripper
    # 固定点の測定とIK検証が済むまで、実機の移動は必ずロックする。
    config["configured"] = False

    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="SO-101 / XIAO 対話式関節校正")
    parser.add_argument("--port", required=True, help="例: /dev/cu.usbmodem1101")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--margin", type=int, default=80,
                        help="関節端点から内側へ取る安全余白tick（既定: 80）")
    parser.add_argument("--gripper-padding", type=int, default=20,
                        help="グリッパー開閉値の外側に取る余白tick（既定: 20）")
    parser.add_argument("--config", type=Path, default=here / "ik" / "fixed_poses.json")
    args = parser.parse_args()

    if not 0 <= args.margin < 1000 or not 0 <= args.gripper_padding < 1000:
        parser.error("marginとgripper-paddingは0以上1000未満にしてください")
    if not args.config.exists():
        parser.error(f"設定ファイルがありません: {args.config}")

    print("SO-101 対話式校正")
    print("- 食器を外し、アームを手で支えてください")
    print("- サーボ電源をすぐ切れる状態にしてください")
    print("- このツールは移動命令を送らず、常にトルクOFFで読み取ります")

    connection: XiaoConnection | None = None
    try:
        wait_to_capture("サーボ電源を入れ、アームの周囲を片付けてください。")
        connection = XiaoConnection(args.port, args.baud)
        connection.torque_off()
        initial = connection.read_joints()
        print("接続OK:", " ".join(f"{i}:{initial[i]}" for i in range(1, 7)))

        wait_to_capture(
            "5つの関節をそれぞれ可動範囲の中央へ置いてください。\n"
            "ベースは正面、腕は無理のない中央姿勢、手首回転は再現しやすい基準向きにします。"
        )
        centers = connection.read_joints()
        print("中央:", " ".join(f"{i}:{centers[i]}" for i in range(1, 6)))

        calibrations: list[dict[str, int]] = []
        for guide in JOINT_GUIDES:
            while True:
                print(f"\n--- ID {guide.servo_id}: {guide.name} ---")
                negative = capture(
                    connection,
                    guide.servo_id,
                    f"この関節だけを負側の安全端までゆっくり動かします。\n方向: {guide.negative_hint}",
                )
                positive = capture(
                    connection,
                    guide.servo_id,
                    f"この関節だけを正側の安全端までゆっくり動かします。\n方向: {guide.positive_hint}",
                )
                try:
                    result = calculate_joint(
                        guide.servo_id, centers[guide.servo_id], negative, positive, args.margin
                    )
                except CalibrationError as exc:
                    print(f"測定エラー: {exc}")
                    if isinstance(exc, WraparoundError):
                        print(
                            "この軸は、物理的な中央をサーボ内部の約2048 tickへ補正すると解決できます。\n"
                            "補正はIDごとに行い、サーボへ永続保存されます。"
                        )
                        action = input(
                            "cで中位補正、Enterで端点を再測定、qで中止: "
                        ).strip().lower()
                        if action == "q":
                            raise KeyboardInterrupt
                        if action == "c":
                            wait_to_capture(
                                f"ID {guide.servo_id}（{guide.name}）を物理的な可動範囲の中央へ戻してください。\n"
                                "この位置がサーボ内部の約2048 tickとして保存されます。"
                            )
                            new_center = connection.calibrate_center(guide.servo_id)
                            centers[guide.servo_id] = new_center
                            print(
                                f"中位補正OK: ID {guide.servo_id} の中央={new_center} tick。"
                                "この軸をもう一度測定します。"
                            )
                            continue
                    retry = input("Enterでこの軸を再測定（qで中止）: ").strip().lower()
                    if retry == "q":
                        raise KeyboardInterrupt
                    continue
                print(
                    f"計算: zero={result['zero_tick']} direction={result['direction']:+d} "
                    f"safe={result['min_tick']}..{result['max_tick']}"
                )
                answer = input("Enterで採用、rで再測定、qで中止: ").strip().lower()
                if answer == "q":
                    raise KeyboardInterrupt
                if answer != "r":
                    calibrations.append(result)
                    break

        while True:
            print("\n--- ID 6: グリッパー ---")
            open_tick = capture(
                connection, 6, "フォークや箸を差し込める、十分に開いた位置へ動かします。"
            )
            closed_tick = capture(
                connection, 6,
                "軽いフォーク等を挟み、落ちないが強く押し続けない閉位置へ動かします。",
            )
            try:
                gripper = calculate_gripper(open_tick, closed_tick, args.gripper_padding)
            except CalibrationError as exc:
                print(f"測定エラー: {exc}")
                retry = input("Enterでグリッパーを再測定（qで中止）: ").strip().lower()
                if retry == "q":
                    raise KeyboardInterrupt
                continue
            break

        print_summary(calibrations, gripper)
        answer = input(
            f"\nEnterで {args.config} へ保存（nで保存せず終了）: "
        ).strip().lower()
        if answer == "n":
            print("保存しませんでした。")
            return 0
        save_config(args.config, calibrations, gripper)
        print(f"保存しました: {args.config}")
        print("configured=falseを維持しています。次は固定点の測定です。")
        return 0
    except KeyboardInterrupt:
        print("\n校正を中止しました。設定ファイルは変更していません。")
        return 130
    except (CalibrationError, OSError, serial.SerialException, json.JSONDecodeError) as exc:
        print(f"校正エラー: {exc}")
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
