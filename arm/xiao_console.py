"""XIAOファームウェアの安全な点検・ティーチング用シリアルCLI。"""
from __future__ import annotations

import argparse
import time

import serial


ALIASES = {
    "ping": "PING",
    "status": "STATUS",
    "read": "READ_JOINTS",
    "torque-off": "TORQUE_OFF",
    "torque-on": "TORQUE_ON",
    "stop": "STOP",
    "home": "HOME",
    "approach-fork": "GOTO APPROACH_FORK",
    "pick-fork": "GOTO PICK_FORK",
    "approach-chopsticks": "GOTO APPROACH_CHOPSTICKS",
    "pick-chopsticks": "GOTO PICK_CHOPSTICKS",
    "approach-place": "GOTO APPROACH_PLACE",
    "place": "GOTO PLACE_UTENSIL",
    "retreat": "GOTO RETREAT",
    "fork": "BRING_UTENSIL manual fork",
    "chopsticks": "BRING_UTENSIL manual chopsticks",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="As Served XIAO console")
    parser.add_argument("command", choices=sorted(ALIASES))
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--listen", type=float, default=1.0,
                        help="送信後に応答を読む秒数。動作完了まで見る場合は20程度")
    args = parser.parse_args()

    with serial.Serial(args.port, args.baud, timeout=0.1) as ser:
        time.sleep(2.0)  # USB CDC再起動待ち
        ser.reset_input_buffer()
        command = ALIASES[args.command]
        ser.write((command + "\n").encode("ascii"))
        ser.flush()
        print(f"> {command}")
        deadline = time.monotonic() + args.listen
        while time.monotonic() < deadline:
            line = ser.readline().decode("ascii", "replace").strip()
            if line:
                print(f"< {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
