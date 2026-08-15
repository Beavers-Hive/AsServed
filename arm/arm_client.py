"""
arm_client.py
SO-101 アーム（XIAO をサーボバスのコントローラとして使う）へ
「この料理に合う食器を運べ」と伝える側。

責務を薄く保つのが要点。ホスト(Mac)側は "どの食器を運ぶか" という意味だけを送り、
逆運動学も軌道もアームの firmware / 上位ライブラリ側に閉じ込める。こうしておくと、
アームが無い環境（--arm mock、あるいはアーム抜きの Tier1 構成）でも投影アプリは
そのまま動く。コンテストの再現性の観点でも、アームを必須にしない構成が重要。

プロトコル（行指向・人間が読める・シリアルモニタでそのまま叩ける）:
    ホスト → XIAO   BRING_UTENSIL <dish_id> <fork|chopsticks>\n
    ホスト → XIAO   HOME\n
    ホスト → XIAO   PING\n
    XIAO → ホスト   OK BRING_UTENSIL <fork|chopsticks>\n / BUSY\n / ERR <reason>\n / PONG\n

フォークと箸のラック位置、提供位置は XIAO 側の固定テーブル
（ティーチング済み姿勢）で持つ。
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class ArmError(RuntimeError):
    pass


class ArmClient:
    """基底。create() でバックエンドを選ぶ。"""

    @staticmethod
    def create(backend: str, port: Optional[str] = None, config: Optional[dict] = None) -> "ArmClient":
        config = config or {}
        if backend == "mock":
            return MockArmClient()
        if backend == "serial":
            return SerialArmClient(port=port or config.get("port"),
                                   baud=int(config.get("baud", 115200)),
                                   timeout_s=float(config.get("timeout_s", 5.0)),
                                   motion_seconds=float(config.get("motion_seconds", 18.0)))
        raise ValueError(f"未知のアームバックエンド: {backend!r}")

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def bring_utensil(self, dish_id: str, utensil: str) -> None:
        raise NotImplementedError

    def home(self) -> None:
        raise NotImplementedError


class MockArmClient(ArmClient):
    """アームが手元に無いときの代役。投影側の開発とデモリハーサルはこれで完結する。"""

    def __init__(self):
        self.log: list = []

    def connect(self) -> None:
        print("[arm/mock] 接続しました（実機なし）")

    def disconnect(self) -> None:
        print("[arm/mock] 切断しました")

    def bring_utensil(self, dish_id: str, utensil: str) -> None:
        line = f"BRING_UTENSIL {dish_id} {utensil}"
        self.log.append(line)
        print(f"[arm/mock] → {line}")

    def home(self) -> None:
        self.log.append("HOME")
        print("[arm/mock] → HOME")


class SerialArmClient(ArmClient):
    """XIAO へシリアルでコマンドを送る。

    送信は非ブロッキングにしない。食器の運搬には十数秒かかるが、その間に投影が
    止まると見栄えが悪いので呼び出しは即戻り、完了・エラー応答はバックグラウンドで監視する。
    """

    def __init__(self, port: Optional[str], baud: int = 115200, timeout_s: float = 5.0,
                 motion_seconds: float = 18.0):
        if not port:
            raise ArmError("シリアルポートが指定されていません（--arm-port か config.arm.port）")
        self.port = port
        self.baud = baud
        self.timeout_s = timeout_s
        self.motion_seconds = motion_seconds
        self._ser = None
        self._busy_until = 0.0
        self._reader_stop = threading.Event()
        self._reader: Optional[threading.Thread] = None

    def connect(self) -> None:
        try:
            import serial  # pyserial
        except ImportError as exc:
            raise ArmError("pyserial が入っていません: uv pip install pyserial") from exc

        self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        time.sleep(2.0)  # XIAO のリセット待ち（USB CDC は開いた瞬間に再起動することがある）
        self._ser.reset_input_buffer()
        self._send("PING")
        reply = self._read_line(timeout_s=3.0)
        if reply != "PONG":
            raise ArmError(f"XIAO から PONG が返りません（受信: {reply!r}）。"
                           f"ポート {self.port} とファームウェアを確認してください。")
        print(f"[arm/serial] 接続しました: {self.port} @ {self.baud}")
        self._reader_stop.clear()
        self._reader = threading.Thread(target=self._drain_replies,
                                        name="xiao-arm-replies", daemon=True)
        self._reader.start()

    def disconnect(self) -> None:
        if self._ser is not None:
            try:
                # 終了時に新しい移動を始めない。実行中の軌道を止め、トルクを切る。
                self._send("STOP")
            finally:
                self._reader_stop.set()
                if self._reader is not None:
                    self._reader.join(timeout=0.5)
                    self._reader = None
                self._ser.close()
                self._ser = None

    def _send(self, line: str) -> None:
        if self._ser is None:
            raise ArmError("未接続です。connect() を先に呼んでください。")
        self._ser.write((line + "\n").encode("ascii"))
        self._ser.flush()

    def _read_line(self, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        buf = b""
        while time.monotonic() < deadline:
            chunk = self._ser.read(64)
            if chunk:
                buf += chunk
                if b"\n" in buf:
                    return buf.split(b"\n")[0].decode("ascii", "replace").strip()
            time.sleep(0.01)
        return ""

    def _drain_replies(self) -> None:
        """XIAOの非同期START/完了/エラー応答を読み、USBバッファを詰まらせない。"""
        ser = self._ser
        if ser is None:
            return
        while not self._reader_stop.is_set():
            try:
                line = ser.readline().decode("ascii", "replace").strip()
            except Exception as exc:
                if not self._reader_stop.is_set():
                    print(f"[arm/serial] 受信を停止しました: {exc}")
                return
            if not line:
                continue
            print(f"[arm/serial] ← {line}")
            if line.startswith("OK BRING_UTENSIL") or line.startswith("OK HOME") \
                    or line.startswith("ERR "):
                self._busy_until = 0.0

    def bring_utensil(self, dish_id: str, utensil: str) -> None:
        # 連打対策。動作中に次を投げても XIAO 側は BUSY を返すだけなので、
        # ホスト側でも最短間隔を守って無駄なコマンドを減らす。
        now = time.monotonic()
        if now < self._busy_until:
            print(f"[arm/serial] 動作中のため無視: {utensil}")
            return
        self._send(f"BRING_UTENSIL {dish_id} {utensil}")
        self._busy_until = now + self.motion_seconds
        print(f"[arm/serial] → BRING_UTENSIL {dish_id} {utensil}")

    def home(self) -> None:
        self._send("HOME")
        self._busy_until = time.monotonic() + self.motion_seconds
