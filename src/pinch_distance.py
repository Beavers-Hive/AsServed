"""
pinch_distance.py
親指と人差し指の **先端どうしの距離** を測って表示する。

`table_sign.py` の指差し入力（`pointer.HandPointer`）と同じ経路
（カメラ → MediaPipe HandLandmarker → 机の mm 座標）を使うが、見るランドマークを
人差し指の先端(8)だけでなく親指の先端(4)にも広げ、2点間の距離を出す。
つまみ幅そのものを入力に使いたい場面（「この大きさ」を指で示す、ピンチで確定する）の
土台であり、まずは距離が安定して取れているかを目で確かめるための道具。

## 距離の出し方は2通りあり、意味が違う

  world : MediaPipe の `hand_world_landmarks`（手の中心を原点とするメートル座標）から
          求めた **実寸の推定値**。カメラ較正が要らず、指が机から浮いていても測れる。
          既定はこちら。手のモデルに基づく推定なので、絶対値は数 mm ずれうる。
  table : 指先をホモグラフィで机の平面へ落としてから測った距離。較正
          （calibration.json + metric.json）が要る代わりに、投影物と同じ土俵の値になる。
          ただし **指が机に触れている前提**。浮くと視差の分だけ伸び縮みする
          （理屈は pointer.py の冒頭を参照）。

両方いつも計算して表示し、`--source` はどちらを「主」の大きな数字にするかだけを決める。

## 値が暴れないようにする仕掛け

指の距離は、指を止めていてもランドマークのノイズで数 mm 揺れる。一方でつまむ動作
そのものは速い。固定係数の EMA ではこの両方を満たせない（弱いと数字が震え、強くすると
つまみに追従しなくなる）ので、2段構えにしている。

  1. **1€ フィルタ**（`--filter euro`, 既定）。推定速度に応じてカットオフを上げる。
     止めているときだけ強く均すので、静止時のちらつきを追従性と引き換えにしない。
     `--min-cutoff` を下げるほど静かになり、`--beta` を上げるほど速い動きに機敏。
  2. **表示側の丸めと不感帯**（`--step-mm` / `--deadband-mm`）。1mm 刻みで丸め、
     1.5mm 以上動くまで数字を書き換えない。下位桁が常に往復するのを止めるのは
     フィルタではなくこちらの仕事。線の描画とつまみ判定には連続値を使い続ける。

生のばらつきは補助行（world / table / px / scale）に丸めずに出しているので、調整中は
そちらを見て `--min-cutoff` を決められる。`--filter none` で素の値とも比べられる。

## 同じ開き具合なのに、机の左右で値が変わるとき

効く順に3つある。補助行（world / table / px / scale / tilt）を見れば、どれかが分かる。

  1. **つまみ軸の向き（tilt）が最大の要因。** 手首が回って親指と人差し指が「上下」に
     並ぶと、開き具合が同じでも数値は大きく縮む。table は机平面への射影長なので
     cos(傾き) 倍そのもの（70度なら 1/3）。world は3次元距離だが、傾くほど距離の大半を
     MediaPipe の最も苦手な z 成分が担うので、やはり崩れる。tilt が 45度を超えたら
     画面に警告を出す。**机の左右どちらでも手の向きを揃えて測ること。**
  2. **モデルのスケール推定の揺れ（scale）。** 手首→中指付け根は開閉しても変わらない
     はずの長さ。これが左右で動くならスケール推定が揺れている。定規で実測して
     `--ref-mm 95` のように渡すと比で打ち消せる。効くのは1〜2割の差まで。
  3. **視差（table のみ）。** ホモグラフィは机の平面上の点にしか使えない。ただし
     この設置（較正値から復元したカメラ高さ ≈ 320mm）では、100mm 浮かせても
     60mm → 約 88mm と **増える** 方向で、左右差はほとんど出ない。数倍の食い違いを
     視差のせいにしないこと。なおレンズ歪みは補正していないので画面端は別途ずれる。

`--log out.csv` で位置・左右（Left/Right）・各距離・tilt を1フレーム1行で残せる。
左で数秒、右で数秒つまんで記録すれば、原因を数字で切り分けられる。

事前準備:
    uv run python src/fetch_model.py                  # 手ランドマークモデル
    （--project / --source table を使うときのみ）
    uv run python src/calibrate.py                    # H_cam→proj
    uv run python src/calibrate_metric.py             # H_table_mm→proj

実行:
    uv run python src/pinch_distance.py               # カメラ映像に重ねて表示
    uv run python src/pinch_distance.py --source table  # 机 mm を主にする
    uv run python src/pinch_distance.py --project     # 机に投影して表示（較正が必要）
    uv run python src/pinch_distance.py --check       # 機材を開かずに設定だけ確認
    uv run python src/pinch_distance.py --min-cutoff 0.4          # もっと数字を静かに
    uv run python src/pinch_distance.py --step-mm 5 --deadband-mm 3   # 5mm 刻みで読む
    uv run python src/pinch_distance.py --filter none --step-mm 0 --deadband-mm 0
                                                      # 素の値をそのまま見る（比較用）
    uv run python src/pinch_distance.py --ref-mm 95   # 手の実測長でスケールを正規化
    uv run python src/pinch_distance.py --log out.csv # 位置・左右つきで記録して切り分け

キーボード操作:
    s        主の数値を world / table で切り替え
    e        English / 日本語（--project のとき）
    d        デバッグ表示（FPS・左右・手の画面位置）
             ※ つまみ軸が 45度以上傾くと、常時その旨を警告表示する
    ESC / q  終了
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import render  # noqa: E402
from geometry import TableFrame  # noqa: E402
from pointer import INDEX_FINGER_TIP, THUMB_TIP, make_hand_landmarker  # noqa: E402

WRIST = 0
MIDDLE_FINGER_MCP = 9

# 表示色（BGR）。投影面では黒＝光を出さない状態なので、明るい色だけを使う。
OPEN = (235, 235, 235)
PINCHED = (120, 230, 160)
MUTED = (120, 120, 120)
ACCENT = (90, 200, 255)
WARN = (60, 190, 255)     # 数値が信用できない状態の注意色（BGR: 橙）

TILT_WARN_DEG = 45.0      # つまみ軸がこれ以上傾いたら、数値が向きの影響を強く受ける

PINCH_CLOSE_MM = 30.0     # これより縮んだら「つまんだ」
PINCH_OPEN_RATIO = 1.35   # 開いたと見なすのは閾値のこの倍。往復のばたつきを止める

MIN_CUTOFF_HZ = 0.7       # 1€ フィルタ: 静止時のカットオフ。下げるほど静かで、動き出しが鈍る
BETA = 0.015              # 1€ フィルタ: 速度に対する追従の強さ。上げるほど速い動きに機敏
STEP_MM = 1.0             # 表示の丸め幅。0 で丸めない
DEADBAND_MM = 1.5         # 表示を書き換える最小変化。丸め幅の半分より必ず大きくする


# --- 平滑化 -----------------------------------------------------------------

def _smoothing_alpha(dt: float, cutoff_hz):
    """カットオフ周波数 cutoff_hz の1次ローパスを、間隔 dt の EMA 係数に直す。

    cutoff_hz は配列でもよい（指先座標のように成分ごとにカットオフが変わるため）。
    """
    tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff_hz, dtype=np.float64))
    return 1.0 / (1.0 + tau / float(dt))


class OneEuroFilter:
    """1€ フィルタ。止めているときは強く均し、動かしているときは追従する。

    指の距離は「止めているつもりでも数 mm 揺れる」一方、つまむ動作そのものは速い。
    固定係数の EMA ではこの2つを同時に満たせない（弱いと数字が震え、強いと遅れる）。
    1€ フィルタは推定速度に応じてカットオフを上げるので、静止時のちらつきだけを消せる。

    min_cutoff を下げると静止時がより静かになり、beta を上げると速い動きに機敏になる。
    スカラーにも配列（指先座標）にも同じものを使う。
    """

    def __init__(self, min_cutoff: float = MIN_CUTOFF_HZ, beta: float = BETA,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or t <= self._t:
            # 初回、または時刻が戻った（＝計測が途切れた）場合は素通し
            self._x, self._dx, self._t = x, np.zeros_like(x), float(t)
            return float(x) if x.ndim == 0 else x.copy()

        dt = float(t) - self._t
        dx = (x - self._x) / dt
        a_d = _smoothing_alpha(dt, self.d_cutoff)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        # 速く動いているほどカットオフを上げる＝遅れを減らす
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = _smoothing_alpha(dt, np.maximum(cutoff, 1e-3))
        self._x = a * x + (1.0 - a) * self._x
        self._t = float(t)
        return float(self._x) if self._x.ndim == 0 else self._x.copy()


class EmaFilter:
    """従来の固定係数 EMA。1€ フィルタと比較する / 挙動を単純に保ちたいとき用。"""

    def __init__(self, alpha: float = 0.45):
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None

    def __call__(self, x, t: float = 0.0):
        x = np.asarray(x, dtype=np.float64)
        self._x = x if self._x is None else self.alpha * x + (1.0 - self.alpha) * self._x
        return float(self._x) if self._x.ndim == 0 else self._x.copy()


class NullFilter:
    """素の値をそのまま返す。生のばらつきを見て調整するための比較用。"""

    def reset(self) -> None:
        pass

    def __call__(self, x, t: float = 0.0):
        x = np.asarray(x, dtype=np.float64)
        return float(x) if x.ndim == 0 else x.copy()


def make_filter(kind: str = "euro", min_cutoff: float = MIN_CUTOFF_HZ, beta: float = BETA,
                ema_alpha: float = 0.45):
    if kind == "euro":
        return OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    if kind == "ema":
        return EmaFilter(alpha=ema_alpha)
    if kind == "none":
        return NullFilter()
    raise ValueError(f"未知のフィルタ種別: {kind!r}")


class StableReadout:
    """**表示する数字だけ** を落ち着かせる。丸めと不感帯で、下位桁の往復を止める。

    フィルタを強くして数字を止めようとすると、つまむ動きへの追従まで鈍る。
    そこで計測値（線の描画やつまみ判定に使う）はフィルタ済みの連続値のまま残し、
    人が読む数字にだけ「step_mm で丸め、deadband_mm を超えるまで書き換えない」を課す。

    deadband_mm は step_mm の半分より大きくすること。等しいと丸めの境目で往復する。
    """

    def __init__(self, step_mm: float = STEP_MM, deadband_mm: float = DEADBAND_MM,
                 decimals: Optional[int] = None):
        self.step_mm = float(step_mm)
        self.deadband_mm = float(deadband_mm)
        self.decimals = (1 if self.step_mm < 1.0 else 0) if decimals is None else int(decimals)
        self.value: Optional[float] = None

    def _quantize(self, mm: float) -> float:
        if self.step_mm <= 0.0:
            return float(mm)
        return round(float(mm) / self.step_mm) * self.step_mm

    def update(self, mm: Optional[float]) -> Optional[float]:
        if mm is None:
            self.value = None
        elif self.value is None or abs(mm - self.value) >= self.deadband_mm:
            self.value = self._quantize(mm)
        return self.value

    def text(self) -> str:
        return "--" if self.value is None else f"{self.value:.{self.decimals}f} mm"


# --- 計測（機材なしで単体テストできる純粋関数） -----------------------------

def _dist(a: Sequence, b: Sequence) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def tips_px_from_landmarks(landmarks, width: int, height: int) -> np.ndarray:
    """正規化ランドマークから (親指先端, 人差し指先端) のカメラ px を (2,2) で返す。"""
    t, i = landmarks[THUMB_TIP], landmarks[INDEX_FINGER_TIP]
    return np.array([[t.x * width, t.y * height],
                     [i.x * width, i.y * height]], dtype=np.float32)


def world_distance_mm(world_landmarks) -> float:
    """`hand_world_landmarks`（メートル）から親指と人差し指の先端の距離を mm で返す。

    原点は手の幾何中心で、カメラからの距離やレンズに依存しない。較正なしで
    「実寸いくつか」を言えるのはこの値だけなので、既定の主表示にしている。
    """
    t, i = world_landmarks[THUMB_TIP], world_landmarks[INDEX_FINGER_TIP]
    return _dist((t.x, t.y, t.z), (i.x, i.y, i.z)) * 1000.0


def hand_scale_mm(world_landmarks) -> float:
    """手首(0) → 中指の付け根(9) の長さ mm。手を開閉しても変わらない剛体長。

    この長さは本来その人の手で一定なので、フレームごとに動いていれば
    それは **モデルが推定した手全体のスケールが揺れている** ということ。
    左右や遠近で数値が変わる原因を切り分ける物差しであり、`--ref-mm` を
    与えたときは「実測値 / この値」で全体を正規化する係数にもなる。
    """
    w, m = world_landmarks[WRIST], world_landmarks[MIDDLE_FINGER_MCP]
    return _dist((w.x, w.y, w.z), (m.x, m.y, m.z)) * 1000.0


def pinch_axis_tilt_deg(world_landmarks) -> float:
    """つまみ軸（親指先端→人差し指先端）が画像平面から何度傾いているか。

    0度 = カメラの画像平面と平行（＝真横から見えている。もっとも精度が出る向き）、
    90度 = カメラをまっすぐ向いている（＝奥行きだけで測ることになり、もっとも苦手）。

    値が場所によって変わる原因の大半はここ。同じ開き具合でも、
      - table は机平面への射影長なので、傾けば cos(傾き) 倍に縮む
      - world は3次元距離だが、傾くほど距離の大半を最も誤差の大きい z 成分が担う
    ため、手首が回るだけで数値が大きく変わる。60度を超えたら向きを疑うこと。
    """
    t, i = world_landmarks[THUMB_TIP], world_landmarks[INDEX_FINGER_TIP]
    d = np.array([i.x - t.x, i.y - t.y, i.z - t.z], dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return 0.0
    return float(np.degrees(np.arcsin(min(abs(d[2]) / n, 1.0))))


def handedness_label(result) -> Optional[str]:
    """'Left' / 'Right'。左右の手で値が違うのかを確かめるために記録する。"""
    hd = getattr(result, "handedness", None)
    if not hd or not hd[0]:
        return None
    return getattr(hd[0][0], "category_name", None)


def table_tips_mm(frame: TableFrame, tips_px: np.ndarray) -> np.ndarray:
    """指先のカメラ px を机の mm 座標へ落とす。指が机に触れている前提。"""
    return frame.cam_to_table(np.asarray(tips_px, dtype=np.float32))


class PinchGate:
    """距離を「つまんだ / 開いた」の2値にする。ヒステリシス付き。

    単純な閾値だと境目で毎フレーム反転して表示がちらつくので、閉じる閾値より
    開く閾値を高くしておく。
    """

    def __init__(self, close_mm: float = PINCH_CLOSE_MM, open_ratio: float = PINCH_OPEN_RATIO):
        self.close_mm = float(close_mm)
        self.open_mm = float(close_mm) * float(open_ratio)
        self.pinched = False

    def update(self, distance_mm: Optional[float]) -> bool:
        if distance_mm is None:
            self.pinched = False
        elif self.pinched:
            self.pinched = distance_mm <= self.open_mm
        else:
            self.pinched = distance_mm <= self.close_mm
        return self.pinched


@dataclasses.dataclass
class PinchSample:
    """1フレーム分の計測結果。指が見えていなければ距離はすべて None。"""

    at: float                                   # time.monotonic()
    tips_px: Optional[np.ndarray] = None        # (2,2) 親指, 人差し指（カメラ px）
    tips_mm: Optional[np.ndarray] = None        # (2,2) 机 mm。較正が無ければ None
    world_mm: Optional[float] = None
    table_mm: Optional[float] = None
    px: Optional[float] = None
    hand_scale_mm: Optional[float] = None   # 手首→中指付け根。モデルのスケール推定の揺れを見る
    handedness: Optional[str] = None        # 'Left' / 'Right'
    tilt_deg: Optional[float] = None        # つまみ軸の傾き。大きいほど数値が信用できない

    @property
    def visible(self) -> bool:
        return self.tips_px is not None

    def value_mm(self, source: str = "world") -> Optional[float]:
        """主表示に使う距離。指定した側が無ければもう一方で代替する。"""
        primary, other = ((self.world_mm, self.table_mm) if source == "world"
                          else (self.table_mm, self.world_mm))
        return primary if primary is not None else other


# --- 取得 -------------------------------------------------------------------

class PinchMeter:
    """カメラ＋HandLandmarker を専用スレッドで回し、最新の計測結果を保持する。

    HandPointer と同じ理由でスレッドに分けている（推論の重さを表示のフレームレートに
    持ち込まない）。表示側から映像も欲しいので、最新フレームも一緒に公開する。
    """

    def __init__(self, camera, model_path: Path, frame: Optional[TableFrame] = None,
                 ema_alpha: float = 0.45, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, lost_timeout_s: float = 0.4,
                 filter_kind: str = "euro", min_cutoff: float = MIN_CUTOFF_HZ,
                 beta: float = BETA, ref_mm: Optional[float] = None):
        self.camera = camera
        self.model_path = Path(model_path)
        self.frame = frame
        self.ema_alpha = float(ema_alpha)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.lost_timeout_s = float(lost_timeout_s)
        self.filter_kind = filter_kind
        # 実測した手首→中指付け根の長さ。与えられていればモデルのスケール推定を毎フレーム補正する
        self.ref_mm = None if ref_mm is None else float(ref_mm)

        # 指先座標と world 距離は別系統で均す。world 距離は指先の px からは作れない
        # （奥行きを含む3次元の量なので）ため、それぞれに専用のフィルタを持たせる。
        def _new():
            return make_filter(filter_kind, min_cutoff=min_cutoff, beta=beta,
                               ema_alpha=self.ema_alpha)
        self._px_filter = _new()
        self._world_filter = _new()
        self._scale_filter = _new()

        self._lock = threading.Lock()
        self._sample = PinchSample(at=0.0)
        self._bgr: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._landmarker = None
        self._mp = None
        self.fps = 0.0

    # --- 準備 --------------------------------------------------------------

    def start(self) -> None:
        if self.frame is not None:
            # cam→table には calibration.json と metric.json の両方が要る。
            # 起動時に一度触って、スレッドの奥ではなくここで失敗させる。
            try:
                _ = self.frame.H_cam_to_table
            except ValueError as exc:
                print(f"[pinch] {exc}")
                print("[pinch] 机 mm での距離は出せません（world の実寸推定のみ表示します）。")
                self.frame = None

        self._landmarker, self._mp = make_hand_landmarker(
            self.model_path, num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.camera.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="PinchMeter", daemon=True)
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

    def measure(self, result, width: int, height: int, now: float) -> PinchSample:
        """推論結果を PinchSample にする。平滑化の状態を持つのでメソッドにしてある。"""
        if not result.hand_landmarks:
            # 手が消えたら状態を捨てる。戻ってきたときに消える前の値へ引っぱられないように。
            self._px_filter.reset()
            self._world_filter.reset()
            self._scale_filter.reset()
            return PinchSample(at=now)

        tips = tips_px_from_landmarks(result.hand_landmarks[0], width, height)
        tips_px = np.asarray(self._px_filter(tips, now), dtype=np.float32)

        world_mm = scale_mm = tilt_deg = None
        world = getattr(result, "hand_world_landmarks", None)
        if world:
            raw = world_distance_mm(world[0])
            tilt_deg = pinch_axis_tilt_deg(world[0])
            scale_mm = float(self._scale_filter(hand_scale_mm(world[0]), now))
            # モデルの全体スケールがずれると、つまみ幅も同じ比率でずれる。実測の
            # 剛体長が分かっていれば、その比で毎フレーム打ち消せる（左右・遠近の差の主因）。
            if self.ref_mm is not None and scale_mm > 1e-6:
                raw *= self.ref_mm / scale_mm
            world_mm = float(self._world_filter(raw, now))

        tips_mm = table_mm = None
        if self.frame is not None:
            tips_mm = table_tips_mm(self.frame, tips_px)
            table_mm = _dist(tips_mm[0], tips_mm[1])

        return PinchSample(at=now, tips_px=tips_px, tips_mm=tips_mm,
                           world_mm=world_mm, table_mm=table_mm,
                           px=_dist(tips_px[0], tips_px[1]),
                           hand_scale_mm=scale_mm, handedness=handedness_label(result),
                           tilt_deg=tilt_deg)

    def _loop(self) -> None:
        t_prev = time.monotonic()
        while not self._stop.is_set():
            try:
                bgr = self.camera.get_frame()
            except Exception as exc:   # カメラの一時的な失敗でアプリ全体を落とさない
                print(f"[pinch] カメラ取得に失敗: {exc!r}")
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            now = time.monotonic()
            result = self._landmarker.detect_for_video(mp_image, int(now * 1000))
            sample = self.measure(result, bgr.shape[1], bgr.shape[0], now)

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst

            with self._lock:
                self._sample = sample
                self._bgr = bgr

    def snapshot(self) -> tuple:
        """(最新の計測, 最新の映像) を返す。手を見失ったまま古い値を出し続けない。"""
        with self._lock:
            sample, bgr = self._sample, self._bgr
        if sample.visible and (time.monotonic() - sample.at) > self.lost_timeout_s:
            sample = PinchSample(at=sample.at)
        return sample, bgr

    def read(self) -> PinchSample:
        return self.snapshot()[0]


# --- 表示 -------------------------------------------------------------------

def tilt_warning(sample: PinchSample, lang: str = "en") -> Optional[str]:
    """つまみ軸が傾きすぎて数値が当てにならないときの警告文。問題なければ None。

    黙って小さい値を出すと「机の左右で値が違う」という形で現れ、原因が
    較正やフィルタに見えてしまう。信用できない向きだと分かるようその場で言う。
    """
    if sample.tilt_deg is None or sample.tilt_deg < TILT_WARN_DEG:
        return None
    shrink = np.cos(np.radians(sample.tilt_deg))
    if lang == "ja":
        return f"指の向きがカメラ寄り（{sample.tilt_deg:.0f}度）。机への射影は約{shrink:.0%}に縮みます"
    return f"axis tilted {sample.tilt_deg:.0f}deg - table value shrinks to ~{shrink:.0%}"


def format_detail(sample: PinchSample) -> str:
    """world / table / px / 手のスケール を1行に並べた補足。

    ここは主表示と違って **丸めも不感帯もかけない**。調整中に生のばらつきが
    どれくらいあるかを見るための行なので、動くこと自体に意味がある。

    scale は手首→中指付け根の長さ。手を左右に動かしてこの値が動くなら、
    world の数値がそのぶん位置によって変わっているということ（原因の切り分け用）。
    """
    def mm(x):
        return "--" if x is None else f"{x:.1f}mm"
    px = "--" if sample.px is None else f"{sample.px:.0f}px"
    hand = sample.handedness or "-"
    tilt = "--" if sample.tilt_deg is None else f"{sample.tilt_deg:.0f}deg"
    return (f"world {mm(sample.world_mm)} | table {mm(sample.table_mm)} | {px} | "
            f"scale {mm(sample.hand_scale_mm)} | tilt {tilt} | {hand}")


class SampleLog:
    """1フレーム1行の CSV。左右・遠近で値が変わる原因を後から数字で確かめるため。

    画面を見ながらでは「どちらが原因か」を言い切れないので、位置・左右・各距離を
    そのまま残す。手が見えているフレームだけ、かつ同じ計測を二重に書かないよう
    `at` が進んだときだけ書く。
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write("t,handedness,cam_x,cam_y,world_mm,table_mm,px,hand_scale_mm,tilt_deg\n")
        self._last_at = None
        self.rows = 0

    def write(self, sample: PinchSample) -> bool:
        if not sample.visible or sample.at == self._last_at:
            return False
        self._last_at = sample.at
        cx, cy = np.asarray(sample.tips_px, dtype=np.float64).mean(axis=0)

        def num(x):
            return "" if x is None else f"{x:.2f}"
        self._fh.write(f"{sample.at:.3f},{sample.handedness or ''},{cx:.1f},{cy:.1f},"
                       f"{num(sample.world_mm)},{num(sample.table_mm)},{num(sample.px)},"
                       f"{num(sample.hand_scale_mm)},{num(sample.tilt_deg)}\n")
        self.rows += 1
        return True

    def close(self) -> None:
        self._fh.close()


def label_anchor(p0: Sequence, p1: Sequence, offset: float,
                 side: Optional[float] = None) -> np.ndarray:
    """線分の中点から線に垂直な向きへ offset 離れた点。

    指を閉じるほど線が短くなるので、中点の真横にラベルを置くと指先の印と重なる。
    垂直に逃がしておけば、距離がいくつでも線とラベルが競合しない。
    side を省略すると y が小さい側（画面の上）を選ぶ。机の mm 座標のように
    y の向きが表示と一致しない空間では、呼ぶ側で side を指定すること。
    """
    a, b = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    mid = (a + b) * 0.5
    d = b - a
    n = float(np.linalg.norm(d))
    perp = np.array([0.0, -1.0]) if n < 1e-6 else np.array([-d[1], d[0]]) / n
    if side is None:
        side = -1.0 if perp[1] > 0 else 1.0
    return mid + perp * (offset * float(side))


def annotate_camera(bgr: np.ndarray, sample: PinchSample, readout: str = "--",
                    source: str = "world", pinched: bool = False,
                    hud: str = "") -> np.ndarray:
    """カメラ映像に指先・距離線・数値を重ねる（その場で書き込み、同じ配列を返す）。

    readout は StableReadout が決めた表示用の文字列。描画側では数値を作らない
    （同じフレーム内で主表示と中点ラベルが違う値になるのを防ぐため）。
    """
    color = PINCHED if pinched else OPEN
    if sample.visible:
        (tx, ty), (ix, iy) = sample.tips_px.astype(int)
        cv2.line(bgr, (tx, ty), (ix, iy), color, 2, cv2.LINE_AA)
        for (x, y), name in (((tx, ty), "thumb"), ((ix, iy), "index")):
            cv2.circle(bgr, (x, y), 9, color, 2, cv2.LINE_AA)
            cv2.putText(bgr, name, (x + 12, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        MUTED, 1, cv2.LINE_AA)
        # 数値は線の脇に置く。指を動かしながら読めるよう、視線を移さずに済む位置。
        (tw, th), _ = cv2.getTextSize(readout, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ax, ay = label_anchor(sample.tips_px[0], sample.tips_px[1], 26.0)
        cv2.putText(bgr, readout, (int(ax - tw * 0.5), int(ay + th * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.putText(bgr, f"{'PINCH' if pinched else 'GAP'}  {readout}",
                (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    cv2.putText(bgr, f"{format_detail(sample)}   [{source}]", (16, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1, cv2.LINE_AA)
    warn = tilt_warning(sample)
    if warn:
        cv2.putText(bgr, warn, (16, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARN, 2, cv2.LINE_AA)
    if hud:
        cv2.putText(bgr, hud, (16, bgr.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    MUTED, 1, cv2.LINE_AA)
    return bgr


def draw_projection(canvas: np.ndarray, frame: TableFrame, sample: PinchSample,
                    bounds_mm, readout: str = "--", source: str = "world",
                    pinched: bool = False, lang: str = "en", hud: str = "") -> None:
    """机に投影する表示。指先の間に実寸の測定線を引き、数値を大きく出す。

    測定線は指のあいだ（机の上）に描くので、投影された線の長さそのものが
    つまみ幅の実寸になる。`--source table` のときは線の長さ＝表示値で一致する。
    """
    x0, y0, x1, y1 = bounds_mm
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    H = y1 - y0
    ja = lang == "ja"
    color = PINCHED if pinched else OPEN

    if sample.tips_mm is not None:
        (tx, ty), (ix, iy) = np.asarray(sample.tips_mm, dtype=np.float64)
        render.draw_polyline_mm(canvas, frame, [[tx, ty], [ix, iy]], color,
                                thickness_px=3, closed=False)
        # 端点の小さな十字。指先のどこを測っているかを見せて、ずれていれば人が直せるようにする。
        for px_, py_ in ((tx, ty), (ix, iy)):
            render.draw_polyline_mm(canvas, frame, [[px_ - 5.0, py_], [px_ + 5.0, py_]],
                                    color, thickness_px=2, closed=False)
            render.draw_polyline_mm(canvas, frame, [[px_, py_ - 5.0], [px_, py_ + 5.0]],
                                    color, thickness_px=2, closed=False)
        # どちら側に逃がすかは机の mm ではなく投影された px で決める。斜め投影では
        # mm の +y が画面の上とは限らず、mm 空間で選ぶとラベルが指先の印に重なる。
        # 逃がす量は文字（高さ12mm）が指先の十字を跨いでも触れない程度に取る
        sides = np.array([label_anchor((tx, ty), (ix, iy), 28.0, s) for s in (1.0, -1.0)])
        on_screen = frame.table_to_proj(sides)
        anchor = sides[0] if on_screen[0][1] < on_screen[1][1] else sides[1]
        render.draw_text_mm(canvas, frame, readout, anchor,
                            height_mm=12.0, color=color, align="center", valign="middle")

    if sample.visible:
        render.draw_text_mm(canvas, frame, readout, (cx, y1 - H * 0.22),
                            height_mm=min(H * 0.16, 54.0), color=color,
                            align="center", valign="bottom")
        label = ("つまみ幅" if ja else "THUMB - INDEX")
        if pinched:
            label = ("つまんでいます" if ja else "PINCHED")
        render.draw_text_mm(canvas, frame, label, (cx, y1 - H * 0.20), height_mm=11.0,
                            color=MUTED, align="center", valign="top")
        render.draw_text_mm(canvas, frame, f"{format_detail(sample)}  [{source}]",
                            (cx, y1 - H * 0.20 + 16.0), height_mm=8.0, color=ACCENT,
                            align="center", valign="top")
        warn = tilt_warning(sample, lang)
        if warn:
            render.draw_text_mm(canvas, frame, warn, (cx, y1 - H * 0.20 + 27.0),
                                height_mm=9.0, color=WARN, align="center", valign="top")
    else:
        render.draw_text_mm(canvas, frame,
                            "カメラに手をかざしてください" if ja else "Show your hand to the camera",
                            (cx, cy), height_mm=14.0, color=MUTED,
                            align="center", valign="middle")

    if hud:
        cv2.putText(canvas, hud, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)


# --- 実行 -------------------------------------------------------------------

def load_frame(metric_path: Path, calibration_path: Path) -> Optional[TableFrame]:
    """較正が揃っていれば TableFrame を返す。無ければ None（world 表示のみになる）。"""
    if not Path(metric_path).exists():
        return None
    return TableFrame.load(Path(metric_path), Path(calibration_path))


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    p = argparse.ArgumentParser(description="親指と人差し指の先端の距離を測って表示する")
    p.add_argument("--config", default=str(root / "config.json"))
    p.add_argument("--metric", default=str(root / "metric.json"))
    p.add_argument("--source", default="world", choices=["world", "table"],
                   help="主表示に使う距離。world=実寸推定（既定） / table=机平面 mm（要較正）")
    p.add_argument("--pinch-mm", type=float, default=None,
                   help=f"「つまんだ」と見なす距離 mm（既定 {PINCH_CLOSE_MM:g}）")
    p.add_argument("--filter", default=None, choices=["euro", "ema", "none"],
                   help="平滑化。euro=1€ フィルタ（既定） / ema=固定係数 / none=生の値")
    p.add_argument("--min-cutoff", type=float, default=None,
                   help=f"1€: 静止時のカットオフ Hz。下げるほど数字が静か（既定 {MIN_CUTOFF_HZ:g}）")
    p.add_argument("--beta", type=float, default=None,
                   help=f"1€: 速い動きへの追従。上げるほど機敏（既定 {BETA:g}）")
    p.add_argument("--step-mm", type=float, default=None,
                   help=f"表示の丸め幅 mm。0 で丸めない（既定 {STEP_MM:g}）")
    p.add_argument("--deadband-mm", type=float, default=None,
                   help=f"表示を書き換える最小変化 mm（既定 {DEADBAND_MM:g}）")
    p.add_argument("--ref-mm", type=float, default=None,
                   help="手首から中指の付け根までの実測 mm。与えるとモデルのスケール推定の"
                        "ずれ（左右・遠近で値が変わる主因）を毎フレーム打ち消す")
    p.add_argument("--log", default=None, metavar="CSV",
                   help="1フレーム1行で位置・左右・各距離を書き出す（原因の切り分け用）")
    p.add_argument("--project", action="store_true",
                   help="カメラ映像ではなくプロジェクターへ投影して表示する（要較正）")
    p.add_argument("--lang", default=None, choices=["en", "ja"],
                   help="--project のときの表示言語。既定は config の table_sign.language")
    p.add_argument("--windowed", action="store_true", help="投影をフルスクリーンにしない")
    p.add_argument("--debug", action="store_true", help="起動時からデバッグ表示を出す")
    p.add_argument("--check", action="store_true", help="機材を開かずに設定だけ確認する")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cam_cfg = cfg.get("camera", {})
    pointer_cfg = cfg.get("pointer", {})
    proj_cfg = cfg.get("projector", {})
    sign_cfg = cfg.get("table_sign", {})
    pinch_cfg = cfg.get("pinch", {})

    # 揺れの調整は現場で何度も回すので、CLI > config > 既定値 の順に効くようにする
    def tune(name, fallback):
        arg = getattr(args, name.replace("-", "_"), None)
        return arg if arg is not None else pinch_cfg.get(name, fallback)

    filter_kind = tune("filter", "euro")
    min_cutoff = float(tune("min_cutoff", MIN_CUTOFF_HZ))
    beta = float(tune("beta", BETA))
    step_mm = float(tune("step_mm", STEP_MM))
    deadband_mm = float(tune("deadband_mm", DEADBAND_MM))
    pinch_mm = float(tune("pinch_mm", PINCH_CLOSE_MM))
    ref_mm = tune("ref_mm", None)
    ref_mm = None if ref_mm is None else float(ref_mm)
    if 0.0 < step_mm and deadband_mm <= step_mm * 0.5:
        # 不感帯が丸め幅の半分以下だと、丸めの境目で表示が往復して直らない
        print(f"[pinch] 警告: deadband_mm({deadband_mm:g}) が step_mm({step_mm:g}) の半分以下です。"
              f"{step_mm * 0.6:g} 以上を推奨します。")

    model = Path(pointer_cfg.get("model_path", "models/hand_landmarker.task"))
    if not model.is_absolute():
        model = root / model
    calibration = root / cfg.get("calibration_file", "calibration.json")
    frame = load_frame(Path(args.metric), calibration)
    lang = args.lang or sign_cfg.get("language", "en")
    source = args.source

    # 較正が無ければ机 mm は出せない。黙って別の値を出すと実寸を誤読するので、はっきり言う。
    if frame is None:
        if args.project:
            print(f"[pinch] {args.metric} がありません。--project には寸法較正が必要です。")
            print("[pinch] src/calibrate.py → src/make_board.py → src/calibrate_metric.py を実行してください。")
            return 1
        if source == "table":
            print(f"[pinch] {args.metric} が無いので --source table は使えません。world に切り替えます。")
            source = "world"

    if args.check:
        print(f"[pinch] モデル = {model} ({'あり' if model.exists() else 'なし'})")
        print(f"[pinch] カメラ = backend={cam_cfg.get('backend', 'avf')} "
              f"name={cam_cfg.get('name')!r} index={cam_cfg.get('index', 0)} "
              f"{cam_cfg.get('width', 1280)}x{cam_cfg.get('height', 720)}")
        if frame is None:
            print("[pinch] 較正 = なし（world の実寸推定のみ）")
        else:
            has_cam = frame.H_cam_to_proj is not None
            rms = ("未較正" if frame.board_rms_mm is None else f"残差RMS {frame.board_rms_mm:.2f} mm")
            print(f"[pinch] 較正 = {rms} / cam→table {'可' if has_cam else '不可（calibration.json なし）'}")
        print(f"[pinch] 主表示 = {source} / つまみ判定 = {pinch_mm:.0f} mm 以下")
        print(f"[pinch] 平滑化 = {filter_kind}"
              + (f"（min_cutoff {min_cutoff:g} Hz / beta {beta:g}）" if filter_kind == "euro" else ""))
        print(f"[pinch] 表示の安定化 = {step_mm:g} mm 刻み / {deadband_mm:g} mm 動くまで書き換えない")
        print(f"[pinch] スケール補正 = "
              + (f"手首→中指付け根を {ref_mm:g} mm として正規化" if ref_mm
                 else "なし（world はモデルの推定スケールのまま）"))
        print(f"[pinch] 表示先 = {'プロジェクター' if args.project else 'カメラ映像ウィンドウ'}")
        return 0 if model.exists() else 1

    from camera import Camera, CameraNotAvailableError  # noqa: E402

    try:
        camera = Camera(index=int(cam_cfg.get("index", 0)),
                        width=int(cam_cfg.get("width", 1280)),
                        height=int(cam_cfg.get("height", 720)),
                        controls=cam_cfg)
    except CameraNotAvailableError as exc:
        print(f"[pinch] カメラを用意できません:\n{exc}")
        return 1

    meter = PinchMeter(camera=camera, model_path=model, frame=frame,
                       ema_alpha=float(pointer_cfg.get("ema_alpha", 0.45)),
                       min_detection_confidence=float(pointer_cfg.get("min_detection_confidence", 0.5)),
                       min_tracking_confidence=float(pointer_cfg.get("min_tracking_confidence", 0.5)),
                       filter_kind=filter_kind, min_cutoff=min_cutoff, beta=beta,
                       ref_mm=ref_mm)
    try:
        meter.start()
    except Exception as exc:
        print(f"[pinch] 計測を開始できません: {exc}")
        return 1
    if meter.frame is None:
        source = "world"

    gate = PinchGate(close_mm=pinch_mm)
    readout = StableReadout(step_mm=step_mm, deadband_mm=deadband_mm)
    log = SampleLog(Path(args.log)) if args.log else None
    show_debug = args.debug
    win = None
    if args.project:
        from projector_window import ProjectorWindow  # noqa: E402

        proj_w, proj_h = int(proj_cfg.get("width", 1920)), int(proj_cfg.get("height", 1080))
        win = ProjectorWindow(display_index=int(proj_cfg.get("display_index", 1)),
                              width=proj_w, height=proj_h,
                              fullscreen=bool(proj_cfg.get("fullscreen", True)) and not args.windowed,
                              mode=proj_cfg.get("mode", "borderless"),
                              above_menu_bar=proj_cfg.get("above_menu_bar"))
        win.open()
        _quad, bounds = meter.frame.projection_bounds_mm(proj_w, proj_h)

    print(f"[pinch] 計測を開始しました。主表示={source} / つまみ判定={pinch_mm:.0f}mm / "
          f"平滑化={filter_kind}。ESC で終了。")
    print("[pinch] 数字が落ち着かないときは --min-cutoff を下げる（例 0.4）、"
          "遅れが気になるときは --beta を上げる（例 0.05）。")

    fps = 0.0
    t_prev = time.monotonic()
    running = True
    try:
        while running:
            sample, bgr = meter.snapshot()
            value = sample.value_mm(source)
            # つまみ判定は連続値で（丸めた表示値で判定すると閾値の前後で粗くなる）、
            # 人が読む数字は StableReadout で別に落ち着かせる。
            pinched = gate.update(value)
            readout.update(value)
            if log is not None:
                log.write(sample)

            now = time.monotonic()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = (1.0 / dt) if fps == 0.0 else 0.9 * fps + 0.1 / dt
            pos = ("--" if not sample.visible
                   else "({:.0f},{:.0f})px".format(*np.asarray(sample.tips_px).mean(axis=0)))
            hud = (f"draw {fps:4.1f}fps | hand {meter.fps:4.1f}fps | source {source} | "
                   f"{sample.handedness or '-'} at {pos}" if show_debug else "")

            if win is not None:
                canvas = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
                draw_projection(canvas, meter.frame, sample, bounds, readout.text(),
                                source, pinched, lang=lang, hud=hud)
                if not win.show(canvas):
                    break
                keys = win.pop_keys()
            else:
                if bgr is None:            # まだ1フレームも来ていない
                    time.sleep(0.02)
                    continue
                view = annotate_camera(bgr.copy(), sample, readout.text(), source,
                                       pinched, hud)
                cv2.imshow("pinch distance", view)
                k = cv2.waitKey(1) & 0xFF
                keys = [] if k == 255 else [{27: "escape", ord("q"): "q"}.get(k, chr(k))]

            for k in keys:
                if k in ("escape", "q"):
                    running = False
                elif k == "s":
                    if meter.frame is None:
                        print("[pinch] 較正が無いので table 表示には切り替えられません。")
                    else:
                        source = "table" if source == "world" else "world"
                        readout.update(None)      # 別系統の値なので持ち越さない
                elif k == "e":
                    lang = "en" if lang == "ja" else "ja"
                elif k == "d":
                    show_debug = not show_debug
    except KeyboardInterrupt:
        print("\n[pinch] Ctrl-C を受け取りました。")
    finally:
        meter.stop()
        if log is not None:
            log.close()
            print(f"[pinch] 計測を {log.rows} 行書き出しました: {args.log}")
        if win is not None:
            win.close()
        else:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
