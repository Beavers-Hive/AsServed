"""
pointer.py
指差しの位置を「机の mm 座標」として供給する。

カメラで人差し指の先端を検出し、既存の較正（H_cam→table）で机の平面へ落とす。
投影側はこの mm 座標だけを見ればよく、カメラの解像度や画角に依存しない。

## 視差について（この方式の一番大事な制約）

ホモグラフィは「点が机の平面上にある」ことを前提にした変換です。指が机から浮いていると、
その分だけ実際の指差し先からずれます。ずれの大きさはカメラの見込み角と浮き高さの積で、
50mm 浮いていれば数十 mm ずれることも普通にあります。

対策は3つ入れてあります。

1. **机に触れて指す前提のUIにする。** 触れていれば指先は本当に平面上にあり、視差はゼロ。
   投影メニューを「触る」動作は、机に映っている以上むしろ自然です。
2. **的を大きくする（最低 70mm 角）。** `ui.MIN_TARGET_MM`。
3. **指先の照準を投影して返す。** 人は自分の指がどう認識されているかを見れば勝手に直します。
   人間を含めた閉ループにするのが、深度センサ無しでの現実解です。

それでも系統的にずれる場合は `config.json` の `pointer.offset_mm` で一定量ずらせます。

## モデルファイル

MediaPipe 1.0 は旧 `mp.solutions` を廃止し、Tasks API + 外部モデルファイルになりました。
リポジトリにはモデルを同梱していません。次で取得してください。

    uv run python src/fetch_model.py
"""
from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from geometry import TableFrame, transform_points

INDEX_FINGER_TIP = 8
THUMB_TIP = 4


def make_hand_landmarker(model_path: Path, num_hands: int = 1,
                         min_detection_confidence: float = 0.5,
                         min_tracking_confidence: float = 0.5):
    """VIDEO モードの HandLandmarker と mediapipe モジュールを (landmarker, mp) で返す。

    指差し（pointer.py）と指の距離計測（pinch_distance.py）で、生成手順も
    「モデルが無い / mediapipe が入っていない」ときの案内も同じにしたいので関数にしてある。
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe が入っていません: uv pip install mediapipe"
        ) from exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(
            f"モデルファイルがありません: {model_path}\n"
            "  uv run python src/fetch_model.py で取得してください。"
        )

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=int(num_hands),
        min_hand_detection_confidence=float(min_detection_confidence),
        min_hand_presence_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    return vision.HandLandmarker.create_from_options(options), mp


@dataclasses.dataclass
class PointerSample:
    point_mm: Optional[np.ndarray]   # 机の mm 座標。見えていなければ None
    at: float                        # time.monotonic()
    raw_cam_px: Optional[np.ndarray] = None


class PointerSource:
    """指差し入力の共通インタフェース。"""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> PointerSample:
        raise NotImplementedError


class NullPointer(PointerSource):
    """指差しを使わない（キーボード操作のみ）。"""

    def read(self) -> PointerSample:
        return PointerSample(point_mm=None, at=time.monotonic())


class ScriptedPointer(PointerSource):
    """機材なしで滞留選択の挙動を確かめるための擬似ポインタ。

    (経過秒, x_mm, y_mm) の並びを線形補間して返す。テストと、投影レイアウトの
    リハーサルに使う。
    """

    def __init__(self, waypoints: Sequence, loop: bool = True):
        self.waypoints = [(float(t), float(x), float(y)) for t, x, y in waypoints]
        if not self.waypoints:
            raise ValueError("waypoints が空です")
        self.loop = loop
        self._t0 = time.monotonic()

    def read(self) -> PointerSample:
        now = time.monotonic()
        return self.at_time(now - self._t0, now)

    def at_time(self, elapsed: float, now: Optional[float] = None) -> PointerSample:
        now = time.monotonic() if now is None else now
        total = self.waypoints[-1][0]
        if self.loop and total > 0:
            elapsed = elapsed % total
        ts = [w[0] for w in self.waypoints]
        xs = [w[1] for w in self.waypoints]
        ys = [w[2] for w in self.waypoints]
        x = float(np.interp(elapsed, ts, xs))
        y = float(np.interp(elapsed, ts, ys))
        return PointerSample(point_mm=np.array([x, y], dtype=np.float32), at=now)


class HandPointer(PointerSource):
    """MediaPipe HandLandmarker で人差し指の先端を追う。

    カメラ取得と推論は専用スレッドで回す。投影ループと同じスレッドで動かすと、
    推論の分だけフレームレートが落ちて投影がカクつくため。
    """

    def __init__(self, frame: TableFrame, camera, model_path: Path,
                 ema_alpha: float = 0.45, offset_mm: Sequence = (0.0, 0.0),
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 lost_timeout_s: float = 0.4):
        self.frame = frame
        self.camera = camera
        self.model_path = Path(model_path)
        self.ema_alpha = float(ema_alpha)
        self.offset_mm = np.array(offset_mm, dtype=np.float32)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.lost_timeout_s = float(lost_timeout_s)

        self._lock = threading.Lock()
        self._sample = PointerSample(point_mm=None, at=0.0)
        self._smoothed: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._landmarker = None
        self.fps = 0.0

    # --- 準備 --------------------------------------------------------------

    def _make_landmarker(self):
        landmarker, self._mp = make_hand_landmarker(
            self.model_path, num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return landmarker

    def start(self) -> None:
        # cam→table は calibration.json と metric.json の両方が要る。ここで一度触っておき、
        # 足りない場合はスレッドの中ではなく起動時に分かるようにする。
        try:
            _ = self.frame.H_cam_to_table
        except ValueError as exc:
            raise RuntimeError(
                f"{exc}\n  指差しにはカメラ較正が必要です。src/calibrate.py を実行してください。"
            ) from exc

        self._landmarker = self._make_landmarker()
        self.camera.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="HandPointer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self.camera.stop()

    # --- 本体 --------------------------------------------------------------

    def _loop(self) -> None:
        import cv2

        t_prev = time.monotonic()
        while not self._stop.is_set():
            try:
                bgr = self.camera.get_frame()
            except Exception as exc:  # カメラの一時的な失敗でアプリ全体を落とさない
                print(f"[pointer] カメラ取得に失敗: {exc!r}")
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            now = time.monotonic()
            result = self._landmarker.detect_for_video(mp_image, int(now * 1000))

            point_mm, cam_px = None, None
            if result.hand_landmarks:
                lm = result.hand_landmarks[0][INDEX_FINGER_TIP]
                h, w = bgr.shape[:2]
                cam_px = np.array([lm.x * w, lm.y * h], dtype=np.float32)
                table = transform_points(self.frame.H_cam_to_table, cam_px[None, :])[0]
                table = table + self.offset_mm

                if self._smoothed is None:
                    self._smoothed = table
                else:
                    a = self.ema_alpha
                    self._smoothed = a * table + (1.0 - a) * self._smoothed
                point_mm = self._smoothed.astype(np.float32)
            else:
                self._smoothed = None

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst

            with self._lock:
                self._sample = PointerSample(point_mm=point_mm, at=now, raw_cam_px=cam_px)

    def read(self) -> PointerSample:
        with self._lock:
            sample = self._sample
        # スレッドが止まった/手が消えたまま古い値を返し続けないようにする
        if sample.point_mm is not None and (time.monotonic() - sample.at) > self.lost_timeout_s:
            return PointerSample(point_mm=None, at=sample.at)
        return sample


def create_pointer(kind: str, frame: TableFrame, camera=None, cfg: Optional[dict] = None,
                   project_root: Optional[Path] = None) -> PointerSource:
    cfg = cfg or {}
    if kind in ("none", None):
        return NullPointer()

    if kind == "scripted":
        return ScriptedPointer(cfg.get("waypoints") or [(0, 0, 0), (1, 100, 100)])

    if kind == "hand":
        if camera is None:
            raise ValueError("hand ポインタにはカメラが必要です")
        root = project_root or Path(__file__).resolve().parent.parent
        model = Path(cfg.get("model_path", "models/hand_landmarker.task"))
        if not model.is_absolute():
            model = root / model
        return HandPointer(
            frame=frame, camera=camera, model_path=model,
            ema_alpha=float(cfg.get("ema_alpha", 0.45)),
            offset_mm=cfg.get("offset_mm", [0.0, 0.0]),
            min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        )

    raise ValueError(f"未知のポインタ種別: {kind!r}")
