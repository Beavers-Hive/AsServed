"""
ui.py
指差しで選ぶための的（ターゲット）と、3秒ホールドで確定する滞留選択。

設計の要点:

- **的はすべて mm で定義する。** 指の位置も mm に変換してから当たり判定するので、
  「指が的の上にあるか」を机の上の物理的な事実として判定できる。画面 px に落として
  判定すると、プロジェクターの解像度や設置角度で挙動が変わってしまう。

- **的の大きさは指の精度から決める。** 指先の位置は、手が机から浮いている分の視差と
  検出のゆらぎで数十 mm ずれる（`pointer.py` 参照）。的は最低でも 70mm 角を確保する。

- **一瞬の検出ロスで進捗を捨てない。** MediaPipe は手が速く動くと数フレーム落ちる。
  grace 期間内なら同じ的の滞留を継続扱いにする。ここを厳しくすると、実際には
  ちゃんと指している人が何度もやり直す羽目になる。

- **一度確定した的は、指が離れるまで再確定しない（ラッチ）。** 指を置きっぱなしにして
  3秒ごとに連続発火するのを防ぐ。
"""
from __future__ import annotations

import dataclasses
import time
from typing import Optional, Sequence

import numpy as np

import render
from geometry import TableFrame, circle_mm, rect_mm

ACCENT = (90, 200, 255)     # BGR: アンバー。サイズ選択（主役）だけに使う
IDLE = (110, 110, 110)
SELECTED = (255, 255, 255)
PROGRESS = (120, 230, 255)
MUTED = (130, 130, 130)
SLICE_ON = (190, 190, 190)  # カット数は選択中でもアンバーにしない（主役を食わないため）
SLICE_OFF = (95, 95, 95)

MIN_TARGET_MM = 70.0        # 指差しで狙える最小の的（実測ベースの下限）


@dataclasses.dataclass
class Target:
    """机の上の的。kind でアプリ側が意味を解釈する。"""

    id: str
    kind: str                  # "dish" | "size" | "order"
    center_mm: tuple
    label: str
    payload: object = None
    shape: str = "rect"        # "rect" | "circle"
    width_mm: float = 100.0
    height_mm: float = 100.0
    diameter_mm: float = 90.0

    def contains(self, p_mm: Sequence) -> bool:
        dx = float(p_mm[0]) - self.center_mm[0]
        dy = float(p_mm[1]) - self.center_mm[1]
        if self.shape == "circle":
            return (dx * dx + dy * dy) <= (self.diameter_mm * 0.5) ** 2
        return abs(dx) <= self.width_mm * 0.5 and abs(dy) <= self.height_mm * 0.5

    def perimeter_mm(self, inflate_mm: float = 0.0, segments: int = 160) -> np.ndarray:
        """外周の点列。滞留の進捗をこの周に沿って伸ばして見せる。"""
        cx, cy = self.center_mm
        if self.shape == "circle":
            return circle_mm(cx, cy, self.diameter_mm + inflate_mm * 2.0, segments=segments)

        quad = rect_mm(cx, cy, self.width_mm + inflate_mm * 2.0,
                       self.height_mm + inflate_mm * 2.0)
        # 4辺を等分して閉じた点列にする（進捗の伸び方を一定速度に見せるため）
        pts = []
        per_side = max(2, segments // 4)
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]
            t = np.linspace(0.0, 1.0, per_side, endpoint=False).reshape(-1, 1)
            pts.append(a + (b - a) * t)
        return np.concatenate(pts, axis=0).astype(np.float32)


@dataclasses.dataclass
class DwellEvent:
    target: Target
    at: float


class DwellSelector:
    """同じ的を dwell_seconds のあいだ指し続けたら確定する。"""

    def __init__(self, dwell_seconds: float = 3.0, grace_seconds: float = 0.35,
                 cooldown_seconds: float = 0.8):
        self.dwell_seconds = float(dwell_seconds)
        self.grace_seconds = float(grace_seconds)
        self.cooldown_seconds = float(cooldown_seconds)

        self.hover: Optional[Target] = None
        self.hover_started_at = 0.0
        self._last_seen_at = 0.0
        self._latched_id: Optional[str] = None   # 確定済み。指が離れるまで再確定しない
        self._cooldown_until = 0.0

    @property
    def progress(self) -> float:
        if self.hover is None or self._latched_id == self.hover.id:
            return 0.0
        elapsed = self._last_seen_at - self.hover_started_at
        return float(np.clip(elapsed / self.dwell_seconds, 0.0, 1.0))

    def reset(self) -> None:
        self.hover = None
        self._latched_id = None

    def update(self, point_mm: Optional[Sequence], targets: Sequence,
               now: Optional[float] = None) -> Optional[DwellEvent]:
        now = time.monotonic() if now is None else now

        found = None
        if point_mm is not None:
            for t in targets:
                if t.contains(point_mm):
                    found = t
                    break

        if found is None:
            # 検出ロスや的の外へ一瞬出ただけなら、grace のあいだは滞留を維持する
            if self.hover is not None and (now - self._last_seen_at) > self.grace_seconds:
                self.hover = None
                self._latched_id = None
            return None

        self._last_seen_at = now
        if self.hover is None or self.hover.id != found.id:
            self.hover = found
            self.hover_started_at = now
            self._latched_id = None
            return None

        if self._latched_id == found.id or now < self._cooldown_until:
            return None

        if (now - self.hover_started_at) >= self.dwell_seconds:
            self._latched_id = found.id
            self._cooldown_until = now + self.cooldown_seconds
            return DwellEvent(target=found, at=now)
        return None


def _aabb(t: Target) -> tuple:
    hw = t.diameter_mm * 0.5 if t.shape == "circle" else t.width_mm * 0.5
    hh = t.diameter_mm * 0.5 if t.shape == "circle" else t.height_mm * 0.5
    return (t.center_mm[0] - hw, t.center_mm[1] - hh, t.center_mm[0] + hw, t.center_mm[1] + hh)


def _rect_overlaps(rect: tuple, others: Sequence) -> bool:
    ax0, ay0, ax1, ay1 = rect
    for t in others:
        bx0, by0, bx1, by1 = _aabb(t)
        if not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0):
            return True
    return False


def _rect_hits_disc(rect: tuple, center: Sequence, radius: float) -> bool:
    """矩形と円が重なるか。実寸プレビューの占有域を守るのに使う。"""
    x0, y0, x1, y1 = rect
    nx = min(max(float(center[0]), x0), x1)
    ny = min(max(float(center[1]), y0), y1)
    dx, dy = nx - float(center[0]), ny - float(center[1])
    return (dx * dx + dy * dy) <= radius * radius


def _preview_reserve(dishes, cfg: dict, pad_mm: float = 20.0) -> float:
    """実寸プレビューが占有する半径(mm)。UI をここへ被せない。

    pad_mm は皿の外に確保する余白。皿まわりの文字は上下にしか出ないので、
    横に並べる的（カット数の列）では小さめの余白で判定する。円で判定している都合上、
    上下向けの余白をそのまま使うと横方向を過剰に空けてしまう。
    """
    reserve = float(cfg.get("preview_reserve_mm", 0.0))
    if reserve > 0.0:
        return reserve
    biggest = max(p.plate_diameter_mm for d in dishes for p in d.portions)
    return biggest * 0.5 + pad_mm


def _column_widths(menu, bounds_mm: tuple, cfg: dict) -> tuple:
    """左右の的の列が使う幅を返す: (料理列, サイズ列, カット列)。カット列は無ければ 0。"""
    x0, y0, x1, y1 = bounds_mm
    margin = float(cfg.get("margin_mm", 24.0))
    W = (x1 - margin) - (x0 + margin)
    tile = max(MIN_TARGET_MM, min(float(cfg.get("dish_tile_mm", 110.0)), W * 0.20))
    circ = max(MIN_TARGET_MM, min(float(cfg.get("size_button_mm", 92.0)), W * 0.17))
    slice_w = 0.0
    if any(d.sliceable for d in menu.dishes):
        slice_d = max(MIN_TARGET_MM, min(float(cfg.get("slice_button_mm", 72.0)), circ * 0.82))
        slice_w = slice_d + float(cfg.get("slice_gap_mm", 8.0))
    return tile, circ, slice_w


def preview_span_mm(menu, bounds_mm: tuple, cfg: Optional[dict] = None) -> tuple:
    """実寸プレビューに使える左右の範囲(mm)を返す。

    皿を投影範囲の真ん中に置くと、左右の的の列と取り合いになる。列が使う幅を先に
    引いてから、その残りの中央に皿を据えるのが正しい順序。皿が机に対して大きいほど
    この差が効く。
    """
    cfg = cfg or {}
    x0, y0, x1, y1 = bounds_mm
    margin = float(cfg.get("margin_mm", 24.0))
    gap = float(cfg.get("preview_gap_mm", 12.0))
    tile, circ, slice_w = _column_widths(menu, bounds_mm, cfg)
    return (x0 + margin + tile + gap, x1 - margin - circ - slice_w - gap)


def build_targets(menu, dish_index: int, bounds_mm: tuple, serving_center_mm: Sequence,
                  cfg: Optional[dict] = None) -> list:
    """投影範囲(mm)に合わせて的を自動配置する。

    投影できる範囲は `TableFrame.projection_bounds_mm()` から取れるので、机の広さや
    プロジェクターの設置距離を設定ファイルに手で書かずに済ませる。
    左に料理、右にサイズ、下に注文ボタン。中央は実寸プレビューのために空ける。
    """
    cfg = cfg or {}
    x0, y0, x1, y1 = bounds_mm
    margin = float(cfg.get("margin_mm", 24.0))
    x0, y0, x1, y1 = x0 + margin, y0 + margin, x1 - margin, y1 - margin
    W, H = x1 - x0, y1 - y0

    tile = max(MIN_TARGET_MM, min(float(cfg.get("dish_tile_mm", 110.0)), W * 0.20))
    circ = max(MIN_TARGET_MM, min(float(cfg.get("size_button_mm", 92.0)), W * 0.17))

    targets: list = []
    dishes = menu.dishes

    # --- 先に下段（注文ボタンの行）を確保する ------------------------------
    # 後回しにすると、料理列が下まで伸びて注文ボタンの置き場が無くなり、
    # 最後は実寸プレビューの上へ追い出される。行の取り合いは先に決着させる。
    order_h = max(MIN_TARGET_MM * 0.8, 62.0)
    order_band_top = y1 - order_h - 12.0

    # --- 左列: 料理 -------------------------------------------------------
    col_x = x0 + tile * 0.5
    col_y0, col_y1 = y0, order_band_top
    col_h = max(tile, col_y1 - col_y0)
    # 品数が増えても上下にはみ出さないよう、料理タイルの高さだけを列内に収める。
    # 横幅はサムネイルの視認性を保ち、縦は指差し可能な範囲で詰める。
    tile_h = min(tile, col_h / max(len(dishes), 1))
    for i, dish in enumerate(dishes):
        cy = (col_y0 + col_y1) * 0.5 + (i - (len(dishes) - 1) * 0.5) * tile_h
        targets.append(Target(
            id=f"dish:{dish.id}", kind="dish", center_mm=(col_x, cy),
            label=dish.name_ja, payload=dish, shape="rect",
            width_mm=tile, height_mm=tile_h,
        ))

    # --- 右列: サイズ -----------------------------------------------------
    # この作品の主役。一番外側・一番大きく置く。手は画面の端へ自然に伸びるので、
    # 端に置くこと自体が「まずここを選ぶもの」という合図になる。
    dish = dishes[dish_index]
    portions = dish.portions
    col_x = x1 - circ * 0.5
    span = min(H, circ * 1.3 * len(portions))
    for i, p in enumerate(portions):
        cy = (y0 + y1) * 0.5 + (i - (len(portions) - 1) * 0.5) * (span / max(len(portions), 1))
        targets.append(Target(
            id=f"size:{p.label}", kind="size", center_mm=(col_x, cy),
            label=p.label, payload=p, shape="circle", diameter_mm=circ,
        ))

    # --- サイズ列の内側: カット数 ------------------------------------------
    # サイズ選択より一段下の情報なので、小さく・内寄りに置く。指せる下限
    # (MIN_TARGET_MM) は割らない。実寸プレビューに重なる場合はカット UI を出さない
    # （出せないより、サイズ選択の見やすさを優先する）。
    if dish.sliceable:
        slice_d = max(MIN_TARGET_MM, min(float(cfg.get("slice_button_mm", 72.0)), circ * 0.82))
        gap = float(cfg.get("slice_gap_mm", 10.0))
        reserve = _preview_reserve(dishes, cfg, pad_mm=8.0)
        span_s = min(H, slice_d * 1.25 * len(dish.slice_options))
        centers = [(y0 + y1) * 0.5 + (i - (len(dish.slice_options) - 1) * 0.5)
                   * (span_s / max(len(dish.slice_options), 1))
                   for i in range(len(dish.slice_options))]

        # サイズ列の内側 → 料理列の内側 の順に試す。どちらも空かなければカット UI は
        # 出さない（無理に置いてサイズ選択の見やすさを損なうより、機能を落とす）。
        for slice_x in (x1 - circ - gap - slice_d * 0.5,
                        x0 + tile + gap + slice_d * 0.5):
            placed = []
            for n, cy in zip(dish.slice_options, centers):
                rect = (slice_x - slice_d * 0.5, cy - slice_d * 0.5,
                        slice_x + slice_d * 0.5, cy + slice_d * 0.5)
                if (rect[0] < x0 or rect[2] > x1
                        or _rect_hits_disc(rect, serving_center_mm, reserve)
                        or _rect_overlaps(rect, targets)):
                    placed = []
                    break
                placed.append(Target(id=f"slice:{n}", kind="slice", center_mm=(slice_x, cy),
                                     label=str(n), payload=n, shape="circle",
                                     diameter_mm=slice_d))
            if placed:
                targets.extend(placed)
                break

    # --- 注文ボタン -------------------------------------------------------
    # 実寸プレビューは机の広さによっては下端近くまで占有するので、料理列の真下
    # （皿から一番遠い下段の隅）を本命にする。そこが埋まっている場合だけ他を試す。
    reserve = _preview_reserve(dishes, cfg)

    cy_order = y1 - order_h * 0.5
    candidates = [
        (x0 + tile * 0.5, min(tile + 20.0, W * 0.26)),         # 料理列の下（本命）
        (x1 - circ * 0.5, min(circ + 20.0, W * 0.26)),         # サイズ列の下
        (float(serving_center_mm[0]), min(200.0, W * 0.32)),   # 下中央
    ]
    for cx, order_w in candidates:
        order_w = max(MIN_TARGET_MM * 1.4, order_w)
        rect = (cx - order_w * 0.5, cy_order - order_h * 0.5,
                cx + order_w * 0.5, cy_order + order_h * 0.5)
        if _rect_hits_disc(rect, serving_center_mm, reserve) or _rect_overlaps(rect, targets):
            continue
        targets.append(Target(id="order", kind="order", center_mm=(cx, cy_order),
                              label="注文する", shape="rect",
                              width_mm=order_w, height_mm=order_h))
        break
    else:
        # どこにも置けないほど狭い。重なりを避けようがないので下中央に戻し、
        # 収まらないことは table_sign --check の警告で知らせる。
        order_w = max(MIN_TARGET_MM * 1.4, min(200.0, W * 0.32))
        targets.append(Target(id="order", kind="order",
                              center_mm=(float(serving_center_mm[0]), cy_order),
                              label="注文する", shape="rect",
                              width_mm=order_w, height_mm=order_h))
    return targets


def draw_column_caption(canvas: np.ndarray, frame: TableFrame, targets: Sequence,
                        kind: str, text: str, height_mm: float, color) -> None:
    """列の一番上の的の上に見出しを置く。

    サイズとカット数が同じ「丸いボタンの列」に見えると序列が伝わらない。見出しの
    大きさと明度でどちらが主かを言い切る。
    """
    items = [t for t in targets if t.kind == kind]
    if not items:
        return
    top = min(items, key=lambda t: t.center_mm[1])
    render.draw_text_mm(canvas, frame, text,
                        (top.center_mm[0], top.center_mm[1] - top.diameter_mm * 0.5 - 5.0),
                        height_mm=height_mm, color=color, align="center", valign="bottom")


def draw_target(canvas: np.ndarray, frame: TableFrame, t: Target, selected: bool,
                progress: float = 0.0, thumbnail: Optional[np.ndarray] = None,
                lang: str = "ja") -> None:
    """的を1つ描く。progress>0 のあいだは外周が伸びて滞留の残りを示す。"""
    color = SELECTED if selected else IDLE
    if t.kind == "slice":
        color = SLICE_ON if selected else SLICE_OFF

    if t.shape == "circle":
        render.draw_polyline_mm(canvas, frame, circle_mm(t.center_mm[0], t.center_mm[1],
                                                         t.diameter_mm, segments=96),
                                color, 3 if selected else 2)
    else:
        quad = rect_mm(t.center_mm[0], t.center_mm[1], t.width_mm, t.height_mm)
        render.draw_polyline_mm(canvas, frame, quad, color, 3 if selected else 2)

    if thumbnail is not None:
        inner = min(t.width_mm, t.height_mm) * 0.66
        render.draw_image_mm(canvas, frame, thumbnail, t.center_mm,
                             width_mm=inner, height_mm=inner,
                             opacity=1.0 if selected else 0.55)
        render.draw_text_mm(canvas, frame, t.label,
                            (t.center_mm[0], t.center_mm[1] + t.height_mm * 0.5 - 6.0),
                            height_mm=11.0, color=color, align="center", valign="bottom")
    elif t.kind == "slice":
        # カット数はサイズの脇役。文字を小さく、選択中でも彩度を落として置く。
        # 中の切り分け線が「何を選んでいるか」を絵で伝えるので、文字は控えめでよい。
        _draw_slice_glyph(canvas, frame, t, selected)
        render.draw_text_mm(canvas, frame, t.label,
                            (t.center_mm[0], t.center_mm[1] + t.diameter_mm * 0.5 - 3.0),
                            height_mm=11.0, color=SLICE_ON if selected else MUTED,
                            align="center", valign="bottom")
    else:
        # サイズは大きく（主役）、それ以外の文字ボタンは控えめに
        height = 26.0 if t.kind == "size" else 15.0
        render.draw_text_mm(canvas, frame, t.label, (t.center_mm[0], t.center_mm[1]),
                            height_mm=height, color=ACCENT if selected else color,
                            align="center", valign="middle",
                            max_width_mm=(t.diameter_mm if t.shape == "circle" else t.width_mm) * 0.86)

    if progress > 0.0:
        ring = t.perimeter_mm(inflate_mm=7.0)
        k = max(2, int(len(ring) * float(np.clip(progress, 0.0, 1.0))))
        render.draw_polyline_mm(canvas, frame, ring[:k], PROGRESS, 5, closed=False)


def _draw_slice_glyph(canvas: np.ndarray, frame: TableFrame, t: Target, selected: bool) -> None:
    """カット数ボタンの中に、その分割数の小さなピザ図を描く。"""
    cx, cy = t.center_mm
    r = t.diameter_mm * 0.30
    color = SLICE_ON if selected else SLICE_OFF
    render.draw_polyline_mm(canvas, frame, circle_mm(cx, cy - t.diameter_mm * 0.08, r * 2.0,
                                                     segments=48),
                            color, 2 if selected else 1)
    n = int(t.payload or 0)
    for i in range(n):
        a = -np.pi / 2 + 2.0 * np.pi * i / n
        render.draw_polyline_mm(
            canvas, frame,
            [[cx, cy - t.diameter_mm * 0.08],
             [cx + r * np.cos(a), cy - t.diameter_mm * 0.08 + r * np.sin(a)]],
            color, 2 if selected else 1, closed=False)


def draw_pointer(canvas: np.ndarray, frame: TableFrame, point_mm: Optional[Sequence],
                 hovering: bool = False) -> None:
    """指先の位置に照準を描く。ユーザーが自分の指がどう認識されているかを見て直せるように。"""
    if point_mm is None:
        return
    color = PROGRESS if hovering else (150, 150, 150)
    render.draw_polyline_mm(canvas, frame,
                            circle_mm(point_mm[0], point_mm[1], 26.0, segments=48),
                            color, 2)
    render.draw_polyline_mm(canvas, frame,
                            circle_mm(point_mm[0], point_mm[1], 5.0, segments=16),
                            color, 3)
