"""
table_sign.py
"As Served" — 注文前に、実際に出てくる量そのままの大きさで料理を机に投影するサイン。

写真のメニューは盛り付け量が分からない。結果として頼みすぎて残す、あるいは足りない。
このサインは料理を **実寸** で机に描き、S/M/L を並べて比べられるようにする。
料理とサイズは机に投影されたメニューを **指差して3秒** で選ぶ。注文を確定すると
SO-101 アームへ料理に合う食器が渡され、パスタにはフォーク、ラーメンには箸が届く。

座標はすべてテーブル平面の mm（`geometry.TableFrame`）。指差しを使わない場合は
カメラも不要で、キーボードだけで動く（撮影時のフォールバックとして残してある）。

事前準備:
    1. uv run python src/calibrate.py                   # H_cam→proj（4隅クリック）
    2. uv run python src/make_board.py                  # 較正ボードを印刷
    3. uv run python src/calibrate_metric.py            # H_table_mm→proj
    4. uv run python src/calibrate_metric.py --verify    # 定規で実寸チェック
    5. uv run python src/fetch_model.py                 # 指差しを使う場合のみ

実行:
    uv run python src/table_sign.py --pointer hand      # 指差しで選ぶ
    uv run python src/table_sign.py                     # キーボードのみ
    uv run python src/table_sign.py --pointer scripted  # 擬似ポインタで挙動確認
    uv run python src/table_sign.py --arm mock          # アームはログのみ
    uv run python src/table_sign.py --lang ja           # 日本語表示

キーボード操作（指差しと併用できる）:
    ← →      料理を切り替え
    ↑ ↓ / 1 2 3   サイズ S/M/L
    ENTER    注文する
    s        カット数を切り替え（4 / 6 / 8）
    c        他サイズの比較リング
    r        100mm スケールバー（実寸の証拠）
    g        アレルゲン表示
    e        English / 日本語
    d        デバッグ表示（ポインタ座標・FPS）
    ESC      終了
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import render  # noqa: E402
import ui  # noqa: E402
from geometry import TableFrame, circle_mm  # noqa: E402
from menu import Menu  # noqa: E402
from pointer import create_pointer  # noqa: E402
from projector_window import ProjectorWindow  # noqa: E402

DIM = (70, 70, 70)
# 比較リングは料理の上に重ねて描くので、机の地色にも明るいピザにも乗る明度にする
COMPARE = (235, 235, 235)
CUT = (225, 225, 225)      # カット線。料理の上に乗るので明るいが、
                           # サイズ比較のリングより目立たせないよう細く薄めにする
ACCENT = (90, 200, 255)
MUTED = (120, 120, 120)
PLATE = (150, 150, 150)
OK = (120, 230, 160)

ORDER_HOLD_S = 6.0     # 注文完了表示を出しておく秒数

UTENSIL_NAMES = {
    "fork": {"ja": "フォーク", "en": "Fork"},
    "chopsticks": {"ja": "箸", "en": "Chopsticks"},
}


def utensil_name(utensil: str, lang: str) -> str:
    """内部識別子を投影用の短い表示名にする。"""
    return UTENSIL_NAMES.get(utensil, {}).get(lang, utensil)


def synthetic_frame(proj_w: int, proj_h: int, span_mm: float = 900.0) -> np.ndarray:
    """較正前のプレビュー用。原点が左上に来る、遠近なしの mm→px 変換。

    実機の較正結果の代わりにはならない。レイアウトと素材の確認にだけ使う。
    """
    s = min(proj_w / span_mm, proj_h / (span_mm * proj_h / proj_w))
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class SignState:
    """表示状態。キー操作・指差し・アームへの指示はここを唯一の真実とする。"""

    def __init__(self, menu: Menu, dish_id: Optional[str], portion_label: Optional[str],
                 lang: str = "en"):
        self.menu = menu
        self.dish_index = menu.index_of(dish_id) if dish_id else 0
        labels = [p.label for p in self.dish.portions]
        self.portion_index = labels.index(portion_label.upper()) if portion_label else min(1, len(labels) - 1)
        self.show_compare = True
        self.show_ruler = False
        self.show_allergens = True
        self.show_debug = False
        self.lang = lang
        # 「8カット」など、料理をまたいで好みを保つ。sliceable でない料理では無視される。
        self.slices = self.dish.slice_options[-1] if self.dish.sliceable else 0
        self.ordered_at: Optional[float] = None

    @property
    def dish(self):
        return self.menu.dishes[self.dish_index]

    @property
    def portion(self):
        return self.dish.portions[self.portion_index]

    @property
    def slice_count(self) -> int:
        """いま選ばれているカット数。カットできない料理では 0。"""
        opts = self.dish.slice_options
        if not opts:
            return 0
        return self.slices if self.slices in opts else opts[-1]

    def set_slices(self, n: int) -> bool:
        if n not in self.dish.slice_options or n == self.slice_count:
            return False
        self.slices = n
        self.ordered_at = None
        return True

    def cycle_slices(self, step: int) -> bool:
        opts = self.dish.slice_options
        if not opts:
            return False
        i = opts.index(self.slice_count)
        return self.set_slices(opts[(i + step) % len(opts)])

    @property
    def ordered(self) -> bool:
        return self.ordered_at is not None and (time.monotonic() - self.ordered_at) < ORDER_HOLD_S

    def set_dish(self, index: int) -> bool:
        index %= len(self.menu.dishes)
        if index == self.dish_index:
            return False
        self.dish_index = index
        self.portion_index = min(self.portion_index, len(self.dish.portions) - 1)
        self.ordered_at = None
        return True

    def set_portion(self, index: int) -> bool:
        if not (0 <= index < len(self.dish.portions)) or index == self.portion_index:
            return False
        self.portion_index = index
        self.ordered_at = None
        return True

    def set_portion_by_label(self, label: str) -> bool:
        labels = [p.label for p in self.dish.portions]
        return self.set_portion(labels.index(label)) if label in labels else False

    def step_portion(self, step: int) -> bool:
        return self.set_portion(int(np.clip(self.portion_index + step, 0,
                                            len(self.dish.portions) - 1)))

    def place_order(self) -> None:
        self.ordered_at = time.monotonic()


TEXT_BAND_MM = 34.0    # 皿の外に文字を置くのに要る帯の高さ（名前16mm + スペック10mm + 余白）


def draw_preview(canvas: np.ndarray, frame: TableFrame, state: SignState, center_mm,
                 bounds_mm=None) -> None:
    """中央の実寸プレビュー: 皿・料理・比較リング・スペック。

    投影範囲の高さが足りないときは、文字を皿の外ではなく**皿のふちの内側**へ回す。
    料理は実寸から動かせない（それが作品の主張なので）ため、譲れるのは文字の置き場所だけ。
    """
    dish, portion = state.dish, state.portion
    cx, cy = float(center_mm[0]), float(center_mm[1])
    ja = state.lang == "ja"

    plate_r = portion.plate_diameter_mm * 0.5
    food_r = portion.food_diameter_mm * 0.5
    compact = True
    if bounds_mm is not None:
        headroom = (cy - plate_r) - bounds_mm[1]
        footroom = bounds_mm[3] - (cy + plate_r)
        compact = headroom < TEXT_BAND_MM or footroom < TEXT_BAND_MM * 0.6

    # --- 選択中の皿と料理（実寸） ----------------------------------------
    render.draw_plate_mm(canvas, frame, (cx, cy), portion.plate_diameter_mm,
                         color=PLATE, thickness_px=3)
    render.draw_image_mm(canvas, frame, dish.image_bgra, (cx, cy),
                         width_mm=portion.food_diameter_mm,
                         height_mm=portion.food_diameter_mm)

    # --- カット線（ピザカッター） ----------------------------------------
    # 実寸のピザの上に実寸で刻むので、「1切れがどのくらいか」もそのまま分かる。
    n = state.slice_count
    if n >= 2:
        for i in range(n):
            a = -np.pi / 2 + 2.0 * np.pi * i / n
            render.draw_polyline_mm(
                canvas, frame,
                [[cx, cy], [cx + food_r * np.cos(a), cy + food_r * np.sin(a)]],
                CUT, thickness_px=2, closed=False)
        render.draw_polyline_mm(canvas, frame, circle_mm(cx, cy, 7.0, segments=16),
                                CUT, thickness_px=2)

    # --- 比較リング: 選んでいないサイズを破線で重ねる -----------------------
    # 「Lにすると皿がここまで大きくなる」を実寸で見せるのがこのサインの核心なので、
    # 料理より **後に** 描く。先に描くと、大きいサイズを選んだとき小さいリングが
    # 料理の下に隠れて、肝心の比較ができなくなる。
    if state.show_compare:
        n = len(dish.portions)
        for i, p in enumerate(dish.portions):
            if i == state.portion_index:
                continue
            render.draw_plate_mm(canvas, frame, (cx, cy), p.food_diameter_mm,
                                 color=COMPARE, thickness_px=2, dashes=28)
            # 直径差が小さいサイズ同士だとラベルを真横に並べると重なるので、
            # サイズごとに角度をずらして置く。
            theta = np.radians(-38.0 + 76.0 * (i / max(n - 1, 1)))
            r = p.food_diameter_mm * 0.5 + 8.0
            render.draw_text_mm(canvas, frame, p.label,
                                (cx + r * np.cos(theta), cy + r * np.sin(theta)),
                                height_mm=12.0, color=COMPARE, align="left", valign="middle")

    # --- スペック表示（箱を置かず1行ずつ。机の上は情報を薄く） --------------
    name = dish.name_ja if ja else dish.name_en
    # 麺類は乾麺の量、それ以外（ピザなど）は直径を出す。ピザで「麺 xxg」と書いても
    # 意味がないうえ、直径こそが客の選ぶ軸なので、そちらを一等地に置く。
    if portion.dry_g is not None:
        amount = f"麺 {portion.dry_g}g" if ja else f"{portion.dry_g}g dry"
    else:
        amount = f"直径 {portion.food_diameter_mm:.0f}mm" if ja else f"{portion.food_diameter_mm:.0f}mm across"
    utensil = utensil_name(dish.utensil, state.lang)
    if ja:
        spec = (f"{portion.label} ・ {amount} ・ {portion.served_g}g ・ "
                f"{portion.kcal:,} kcal ・ {utensil}")
        price = f"¥{portion.price_yen:,}"
    else:
        spec = (f"{portion.label}  ·  {amount}  ·  {portion.served_g} g  ·  "
                f"{portion.kcal:,} kcal  ·  {utensil}")
        price = f"JPY {portion.price_yen:,}"

    if compact:
        # 料理と皿のふちのあいだの帯の**中央**に入れる。皿の縁にラベルが載っているように
        # 見えるうえ、帯からはみ出さないので投影範囲の外へ逃げない。文字高さも帯に収める。
        band = max(6.0, plate_r - food_r)
        band_center = (food_r + plate_r) * 0.5
        name_y, name_h, name_v = cy - band_center, min(14.0, band - 3.0), "middle"
        spec_y, spec_h, spec_v = cy + band_center, min(10.0, band - 3.0), "middle"
        allergen_y = cy + plate_r + 6.0
    else:
        name_y, name_h, name_v = cy - plate_r - 16.0, 14.0, "bottom"
        spec_y, spec_h, spec_v = cy + plate_r + 6.0, 10.0, "top"
        allergen_y = spec_y + 13.0

    inner_w = portion.food_diameter_mm * 1.15
    render.draw_text_mm(canvas, frame, f"{name}   {price}", (cx, name_y),
                        height_mm=name_h, color=(255, 255, 255), align="center",
                        valign=name_v, max_width_mm=inner_w)
    render.draw_text_mm(canvas, frame, spec, (cx, spec_y), height_mm=spec_h,
                        color=(210, 210, 210), align="center", valign=spec_v,
                        max_width_mm=inner_w)

    if state.show_allergens:
        allergens = dish.allergens_ja if ja else dish.allergens_en
        if allergens:
            head = "アレルゲン: " if ja else "Allergens: "
            render.draw_text_mm(canvas, frame, head + " / ".join(allergens),
                                (cx, allergen_y), height_mm=9.0, color=(120, 190, 255),
                                align="center", valign="top")

    # --- 実寸の証拠となるスケールバー ------------------------------------
    if state.show_ruler:
        bar_y = cy + plate_r + (26.0 if compact else 34.0)
        if bounds_mm is not None:
            bar_y = min(bar_y, bounds_mm[3] - 14.0)
        x0 = cx - 50.0
        render.draw_polyline_mm(canvas, frame, [[x0, bar_y], [x0 + 100.0, bar_y]],
                                color=(255, 255, 255), thickness_px=3, closed=False)
        for x in (x0, x0 + 50.0, x0 + 100.0):
            render.draw_polyline_mm(canvas, frame, [[x, bar_y - 4.0], [x, bar_y + 4.0]],
                                    color=(255, 255, 255), thickness_px=3, closed=False)
        render.draw_text_mm(canvas, frame, "100 mm", (cx, bar_y + 6.0), height_mm=8.0,
                            color=(255, 255, 255), align="center", valign="top")


CONFIRM_FILL = (70, 185, 250)      # BGR: 温かいアンバー。机が一面これになる
CONFIRM_INK = (28, 24, 20)         # その上に載せる文字（＝投影しない部分）
CONFIRM_GROW_S = 0.45              # 中心から広がりきるまでの秒数


def draw_order_confirmation(canvas: np.ndarray, frame: TableFrame, state: SignState,
                            bounds_mm, center_mm) -> None:
    """注文が通ったことを投影面いっぱいで知らせる。

    机の上では黒＝光を出さない状態なので、逆に面を明るく塗りつぶすと「机全体が
    反応した」ように見える。文字は塗りの上に暗色で置く（＝そこだけ光を抜く）。
    """
    x0, y0, x1, y1 = bounds_mm
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    W, H = x1 - x0, y1 - y0
    ja = state.lang == "ja"

    elapsed = time.monotonic() - (state.ordered_at or 0.0)
    grow = float(np.clip(elapsed / CONFIRM_GROW_S, 0.0, 1.0))
    grow = 1.0 - (1.0 - grow) ** 3          # ease-out。最後だけゆっくり止まる
    full_r = float(np.hypot(W, H)) * 0.5 + 20.0
    render.fill_poly_mm(canvas, frame, circle_mm(cx, cy, full_r * 2.0 * grow, segments=192),
                        CONFIRM_FILL)
    if grow < 0.35:
        return                               # 広がりきる前は文字を出さない

    inner = W * 0.86
    # チェックマーク。見出しと縦に噛み合わないよう、十分に間隔を空けて上に置く。
    s = min(W, H) * 0.065
    check_cy = cy - H * 0.18
    check = np.array([[cx - s * 1.1, check_cy],
                      [cx - s * 0.25, check_cy + s * 0.85],
                      [cx + s * 1.25, check_cy - s * 0.9]], dtype=np.float32)
    render.draw_polyline_mm(canvas, frame, check, CONFIRM_INK,
                            thickness_px=max(6, int(s * 0.9)), closed=False)

    render.draw_text_mm(canvas, frame, "ご注文ありがとうございます" if ja else "ORDER PLACED",
                        (cx, cy + H * 0.04), height_mm=min(H * 0.13, 46.0), color=CONFIRM_INK,
                        align="center", valign="bottom", max_width_mm=inner)

    dish, portion = state.dish, state.portion
    name = dish.name_ja if ja else dish.name_en
    price = f"¥{portion.price_yen:,}" if ja else f"JPY {portion.price_yen:,}"
    render.draw_text_mm(canvas, frame, f"{name}  {portion.label}  ·  {price}",
                        (cx, cy + H * 0.08), height_mm=min(H * 0.075, 26.0), color=CONFIRM_INK,
                        align="center", valign="top", max_width_mm=inner)

    size = (f"直径 {portion.food_diameter_mm:.0f}mm ・ {portion.served_g}g ・ {portion.kcal:,} kcal"
            if ja else
            f"{portion.food_diameter_mm:.0f} mm across · {portion.served_g} g · {portion.kcal:,} kcal")
    if state.slice_count >= 2:
        size += (f" ・ {state.slice_count}カット" if ja
                 else f" · cut into {state.slice_count}")
    render.draw_text_mm(canvas, frame, size, (cx, cy + H * 0.20), height_mm=min(H * 0.038, 13.0),
                        color=CONFIRM_INK, align="center", valign="top", max_width_mm=inner * 0.8)

    utensil = utensil_name(dish.utensil, state.lang)
    render.draw_text_mm(canvas, frame,
                        (f"ロボットが{utensil}をお持ちします" if ja
                         else f"Robot is bringing your {utensil.lower()}"),
                        (cx, cy + H * 0.28), height_mm=min(H * 0.032, 11.0), color=CONFIRM_INK,
                        align="center", valign="top", max_width_mm=inner * 0.7)

    # 残り時間。いつメニューへ戻るかが見えていれば、客は待てばよいと分かる
    remain = 1.0 - float(np.clip(elapsed / ORDER_HOLD_S, 0.0, 1.0))
    bar_w = W * 0.5
    bar_y = y1 - H * 0.06
    if remain > 0.0:
        render.draw_polyline_mm(canvas, frame,
                                [[cx - bar_w * 0.5, bar_y],
                                 [cx - bar_w * 0.5 + bar_w * remain, bar_y]],
                                CONFIRM_INK, thickness_px=4, closed=False)


def draw_frame(canvas: np.ndarray, frame: TableFrame, state: SignState, targets,
               selector: ui.DwellSelector, point_mm, center_mm, hud: str = "",
               bounds_mm=None) -> None:
    ja = state.lang == "ja"

    # 注文が通ったら投影面を全部使って知らせる。メニューは一旦引っ込める。
    if state.ordered and bounds_mm is not None:
        draw_order_confirmation(canvas, frame, state, bounds_mm, center_mm)
        if hud:
            cv2.putText(canvas, hud, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 1, cv2.LINE_AA)
        return
    hover_id = selector.hover.id if selector.hover is not None else None

    for t in targets:
        if t.kind == "dish":
            selected = t.payload is state.dish
            thumb = t.payload.image_bgra
        elif t.kind == "size":
            selected = t.payload.label == state.portion.label
            thumb = None
        elif t.kind == "slice":
            selected = int(t.payload) == state.slice_count
            thumb = None
        else:
            selected = state.ordered
            thumb = None

        label_override = None
        if t.kind == "dish" and not ja:
            label_override = t.payload.name_en
        elif t.kind == "order":
            label_override = ("ご注文ありがとうございました" if ja else "Order placed") \
                if state.ordered else ("注文する" if ja else "Order")

        drawn = t if label_override is None else _relabel(t, label_override)
        ui.draw_target(canvas, frame, drawn, selected=selected,
                       progress=selector.progress if hover_id == t.id else 0.0,
                       thumbnail=thumb, lang=state.lang)

    # 列の見出し。同じ丸ボタンの列が2本あると序列が伝わらないので、大きさと明度で
    # 「サイズが主、カット数が従」であることを言い切る。
    ui.draw_column_caption(canvas, frame, targets, "size",
                           "サイズ" if ja else "SIZE", 13.0, ui.ACCENT)
    ui.draw_column_caption(canvas, frame, targets, "slice",
                           "カット数" if ja else "SLICES", 8.5, ui.SLICE_OFF)

    # 1切れの重さ。カット数の脇に置くことで、この機能もサイズの話（実寸・量）に接続する。
    # サイズを上げれば同じカット数でも1切れが大きくなる、が数字で見える。
    slice_targets = [t for t in targets if t.kind == "slice"]
    if slice_targets and state.slice_count >= 2:
        bottom = max(slice_targets, key=lambda t: t.center_mm[1])
        per = state.portion.served_g / state.slice_count
        note = f"1切れ 約{per:.0f}g" if ja else f"{per:.0f} g per slice"
        render.draw_text_mm(canvas, frame, note,
                            (bottom.center_mm[0],
                             bottom.center_mm[1] + bottom.diameter_mm * 0.5 + 5.0),
                            height_mm=8.5, color=ui.SLICE_OFF, align="center", valign="top")

    draw_preview(canvas, frame, state, center_mm, bounds_mm)

    if bounds_mm is not None:
        hint = ("指を3秒置くと選べます" if ja else "Point and hold for 3 seconds")
        render.draw_text_mm(canvas, frame, hint,
                            (float(bounds_mm[0]) + 6.0, float(bounds_mm[3]) - 6.0),
                            height_mm=8.0, color=MUTED, align="left", valign="bottom")

    ui.draw_pointer(canvas, frame, point_mm, hovering=hover_id is not None)

    if hud:
        cv2.putText(canvas, hud, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)


def _relabel(t: ui.Target, label: str) -> ui.Target:
    import dataclasses
    return dataclasses.replace(t, label=label)


def apply_selection(event: ui.DwellEvent, state: SignState, arm) -> None:
    """滞留で確定した的を状態へ反映する。"""
    t = event.target
    if t.kind == "dish":
        state.set_dish(state.menu.index_of(t.payload.id))
    elif t.kind == "size":
        state.set_portion_by_label(t.payload.label)
    elif t.kind == "slice":
        state.set_slices(int(t.payload))
    elif t.kind == "order":
        submit_order(state, arm)


def submit_order(state: SignState, arm) -> None:
    """注文を確定し、料理に合う食器をアームへ1回だけ指示する。"""
    state.place_order()
    print(f"[table_sign] 注文: {state.dish.id} {state.portion.label} "
          f"¥{state.portion.price_yen:,} / 食器={state.dish.utensil}")
    if arm is not None:
        arm.bring_utensil(state.dish.id, state.dish.utensil)


def handle_keys(keys, state: SignState, arm) -> bool:
    """押されたキーを状態へ反映する。終了要求で False を返す。"""
    for k in keys:
        if k == "right":
            state.set_dish(state.dish_index + 1)
        elif k == "left":
            state.set_dish(state.dish_index - 1)
        elif k == "up":
            state.step_portion(-1)
        elif k == "down":
            state.step_portion(+1)
        elif k in ("1", "2", "3", "4"):
            state.set_portion(int(k) - 1)
        elif k in ("return", "enter"):
            submit_order(state, arm)
        elif k == "s":
            state.cycle_slices(+1)
        elif k == "c":
            state.show_compare = not state.show_compare
        elif k == "r":
            state.show_ruler = not state.show_ruler
        elif k == "g":
            state.show_allergens = not state.show_allergens
        elif k == "e":
            state.lang = "en" if state.lang == "ja" else "ja"
        elif k == "d":
            state.show_debug = not state.show_debug
        elif k == "q":
            return False

    return True


def resolve_serving_center(cfg_value, bounds_mm, order_h: float = 70.0,
                           plate_radius_mm: float = 0.0, span_mm=None):
    """提供位置。設定があればそれを使い、無ければ投影範囲から自動で決める。

    アームの PLACE 姿勢と一致させる必要があるので、実機では config に固定値を入れること。

    自動のときは
      - 横: 左右の的の列を除いた残りの中央（`ui.preview_span_mm`）
      - 縦: 一番大きい皿が投影範囲からはみ出さない位置へ寄せる
    とする。投影範囲の真ん中に置くと、皿が大きいときに列と取り合い、
    注文ボタン分の下寄せがそのまま「皿の上端が切れる」ことになる。
    """
    if cfg_value:
        return [float(cfg_value[0]), float(cfg_value[1])]
    x0, y0, x1, y1 = bounds_mm
    if span_mm is not None and span_mm[1] > span_mm[0]:
        cx = (float(span_mm[0]) + float(span_mm[1])) * 0.5
    else:
        cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5 - order_h * 0.35
    r = float(plate_radius_mm)
    if r > 0.0 and (y1 - y0) >= 2.0 * r:
        cy = float(np.clip(cy, y0 + r, y1 - r))
    return [cx, cy]


def main() -> int:
    root = Path(__file__).resolve().parent.parent

    p = argparse.ArgumentParser(description="As Served — 実寸の卓上メニューサイン")
    p.add_argument("--config", default=str(root / "config.json"))
    p.add_argument("--menu", default=str(root / "menu.json"))
    p.add_argument("--metric", default=str(root / "metric.json"))
    p.add_argument("--dish", default=None, help="起動時に表示する料理 id")
    p.add_argument("--portion", default=None, help="起動時のサイズ S/M/L")
    p.add_argument("--lang", default=None, choices=["en", "ja"],
                   help="表示言語。既定は config の table_sign.language（en）")
    p.add_argument("--pointer", default=None, choices=["none", "hand", "scripted"],
                   help="指差し入力。既定は config の pointer.kind")
    p.add_argument("--dwell", type=float, default=None, help="確定までの秒数（既定 3.0）")
    p.add_argument("--arm", default=None, choices=["mock", "serial"],
                   help="SO-101 アーム連携。mock はログ出力のみ")
    p.add_argument("--arm-port", default=None, help="--arm serial のときのシリアルポート")
    p.add_argument("--windowed", action="store_true", help="フルスクリーンにしない")
    p.add_argument("--check", action="store_true", help="機材を開かずに設定と素材だけ検証する")
    p.add_argument("--preview", default=None, metavar="PNG",
                   help="投影せずに1フレームを PNG に書き出す（レイアウト確認用）")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    sign_cfg = cfg.get("table_sign", {})
    pointer_cfg = cfg.get("pointer", {})
    proj_cfg = cfg.get("projector", {})
    proj_w, proj_h = int(proj_cfg.get("width", 1920)), int(proj_cfg.get("height", 1080))

    menu = Menu.load(Path(args.menu), project_root=root)
    problems = menu.validate()
    for msg in problems:
        print(f"[table_sign] メニュー警告: {msg}")

    lang = args.lang or sign_cfg.get("language", "en")
    state = SignState(menu, args.dish, args.portion, lang=lang)
    font = render.find_font()
    if font is None:
        print("[table_sign] 警告: 日本語フォントが見つかりません。英数字のみの簡易描画になります。")
        print("[table_sign]   環境変数 AS_SERVED_FONT にフォントファイルのパスを指定できます。")

    metric_path = Path(args.metric)
    calibrated = metric_path.exists()
    if calibrated:
        frame = TableFrame.load(metric_path, root / cfg.get("calibration_file", "calibration.json"))
    elif args.check or args.preview:
        # 較正前でもレイアウトと素材は確認したいので、遠近なしの仮の変換で代用する。
        print(f"[table_sign] {metric_path} が無いので仮の寸法変換で表示します（実機では要較正）。")
        frame = TableFrame(H_table_to_proj=synthetic_frame(proj_w, proj_h))
    else:
        print(f"[table_sign] {metric_path} がありません。")
        print("[table_sign] 先に src/calibrate.py → src/make_board.py → src/calibrate_metric.py を実行してください。")
        return 1

    _quad, bounds = frame.projection_bounds_mm(proj_w, proj_h)
    biggest_plate = max(p.plate_diameter_mm for d in menu.dishes for p in d.portions)
    center_mm = resolve_serving_center(sign_cfg.get("serving_center_mm"), bounds,
                                       plate_radius_mm=biggest_plate * 0.5,
                                       span_mm=ui.preview_span_mm(menu, bounds, sign_cfg))
    dwell_s = args.dwell if args.dwell is not None else float(sign_cfg.get("dwell_seconds", 3.0))
    selector = ui.DwellSelector(dwell_seconds=dwell_s,
                                grace_seconds=float(sign_cfg.get("dwell_grace_seconds", 0.35)))

    def targets_now():
        return ui.build_targets(menu, state.dish_index, bounds, center_mm, sign_cfg)

    if args.check:
        print(f"[table_sign] メニュー {len(menu.dishes)} 品 / 問題 {len(problems)} 件")
        print(f"[table_sign] 寸法較正 = "
              f"{f'残差RMS {frame.board_rms_mm:.2f} mm' if frame.board_rms_mm is not None else '未較正'}")
        print(f"[table_sign] 投影範囲 = x {bounds[0]:.0f}..{bounds[2]:.0f} mm / "
              f"y {bounds[1]:.0f}..{bounds[3]:.0f} mm "
              f"({bounds[2] - bounds[0]:.0f} x {bounds[3] - bounds[1]:.0f} mm)")
        print(f"[table_sign] 提供位置 = [{center_mm[0]:.0f}, {center_mm[1]:.0f}] mm / "
              f"局所倍率 = {frame.px_per_mm_at(center_mm[0], center_mm[1]):.2f} px/mm")
        print(f"[table_sign] 滞留確定 = {dwell_s:.1f} 秒 / フォント = {font}")

        # 実寸の皿が投影範囲に収まるかは、机の広さとプロジェクターの設置で決まる。
        # 収まらない場合は投影を大きくするしかない（料理を縮めたら作品の意味がなくなる）。
        biggest = biggest_plate
        avail_h = bounds[3] - bounds[1]
        headroom = (center_mm[1] - biggest * 0.5) - bounds[1]
        footroom = bounds[3] - (center_mm[1] + biggest * 0.5)
        if min(headroom, footroom) < 0:
            print(f"[table_sign] 警告: 最大の皿 φ{biggest:.0f}mm が投影範囲（高さ {avail_h:.0f}mm）に"
                  f"収まりません。プロジェクターを離すか、投影を大きくしてください。")
        elif headroom < TEXT_BAND_MM or footroom < TEXT_BAND_MM * 0.6:
            print(f"[table_sign] 情報: 高さの余裕が少ないため（上 {headroom:.0f}mm / 下 {footroom:.0f}mm）、"
                  f"料理名とスペックを皿のふちの内側に描きます。")
        for t in targets_now():
            size = (f"φ{t.diameter_mm:.0f}mm" if t.shape == "circle"
                    else f"{t.width_mm:.0f}x{t.height_mm:.0f}mm")
            print(f"[table_sign]   的 {t.id:<18} {size:>12} @ "
                  f"({t.center_mm[0]:.0f}, {t.center_mm[1]:.0f})")
        for d in menu.dishes:
            sizes = ", ".join(f"{x.label}:{x.food_diameter_mm:g}mm/{x.served_g}g" for x in d.portions)
            print(f"[table_sign]   {d.id}: {sizes}")
        return 0 if not problems else 1

    if args.preview:
        canvas = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
        draw_frame(canvas, frame, state, targets_now(), selector, None, center_mm,
                   bounds_mm=bounds)
        cv2.imwrite(args.preview, canvas)
        print(f"[table_sign] プレビューを書き出しました: {args.preview}"
              f"（{'較正済み' if calibrated else '仮変換'}）")
        return 0

    # --- 入力（指差し） ---------------------------------------------------
    pointer_kind = args.pointer or pointer_cfg.get("kind", "none")
    camera = None
    if pointer_kind == "hand":
        from camera import Camera, CameraNotAvailableError  # noqa: E402

        cam_cfg = cfg.get("camera", {})
        try:
            camera = Camera(index=int(cam_cfg.get("index", 0)),
                            width=int(cam_cfg.get("width", 1280)),
                            height=int(cam_cfg.get("height", 720)),
                            controls=cam_cfg)
        except CameraNotAvailableError as e:
            print(f"[table_sign] カメラを用意できません:\n{e}")
            return 1
    if pointer_kind == "scripted" and not pointer_cfg.get("waypoints"):
        # 的の上を順に通る既定の軌跡を自動生成しておく（レイアウト確認用）
        wp, t = [], 0.0
        for target in targets_now():
            wp.append((t, target.center_mm[0], target.center_mm[1]))
            t += dwell_s + 1.5
            wp.append((t, target.center_mm[0], target.center_mm[1]))
            t += 0.8
        pointer_cfg = dict(pointer_cfg, waypoints=wp)

    try:
        src = create_pointer(pointer_kind, frame, camera=camera, cfg=pointer_cfg,
                             project_root=root)
        src.start()
    except Exception as exc:
        print(f"[table_sign] 指差し入力を開始できません: {exc}")
        return 1

    arm = None
    if args.arm:
        sys.path.insert(0, str(root / "arm"))
        from arm_client import ArmClient  # noqa: E402

        arm = ArmClient.create(args.arm, port=args.arm_port, config=cfg.get("arm", {}))
        arm.connect()

    win = ProjectorWindow(display_index=int(proj_cfg.get("display_index", 1)),
                          width=proj_w, height=proj_h,
                          fullscreen=bool(proj_cfg.get("fullscreen", True)) and not args.windowed,
                          mode=proj_cfg.get("mode", "borderless"),
                          above_menu_bar=proj_cfg.get("above_menu_bar"))
    win.open()
    print(f"[table_sign] 投影を開始しました。ポインタ={pointer_kind} / 確定={dwell_s:.1f}秒。ESC で終了。")
    fps = 0.0
    t_prev = time.monotonic()
    try:
        while True:
            if not handle_keys(win.pop_keys(), state, arm):
                break

            targets = targets_now()
            sample = src.read()
            event = selector.update(sample.point_mm, targets)
            if event is not None:
                apply_selection(event, state, arm)

            now = time.monotonic()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = (1.0 / dt) if fps == 0.0 else 0.9 * fps + 0.1 / dt

            hud = ""
            if state.show_debug:
                pos = ("--" if sample.point_mm is None
                       else f"({sample.point_mm[0]:.0f}, {sample.point_mm[1]:.0f})mm")
                pfps = getattr(src, "fps", 0.0)
                hud = (f"draw {fps:4.1f}fps | pointer {pfps:4.1f}fps {pos} | "
                       f"hover {selector.hover.id if selector.hover else '-'} "
                       f"{selector.progress * 100:3.0f}%")

            canvas = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
            draw_frame(canvas, frame, state, targets, selector, sample.point_mm, center_mm,
                       hud, bounds_mm=bounds)
            if not win.show(canvas):
                break
    except KeyboardInterrupt:
        print("\n[table_sign] Ctrl-C を受け取りました。")
    finally:
        src.stop()
        win.close()
        if arm is not None:
            arm.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
