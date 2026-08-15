"""
render.py
テーブル平面の mm 座標で図と文字を描き、プロジェクター px へ透視変換して合成する。

方針: 描画APIの引数はすべて mm。px を直接触るのはこのファイルの中だけにする。
文字も「高さ何 mm」で指定する。文字を一度オフスクリーンに描いてから mm 矩形として
ワープするので、机を斜めから投影していても文字が正しく台形補正される。
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Sequence

import cv2
import numpy as np

from geometry import TableFrame, circle_mm, rect_mm

# macOS 標準で入っている日本語フォントを優先的に探す。
FONT_CANDIDATES = (
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
)

WHITE = (255, 255, 255)


def find_font() -> Optional[str]:
    env = os.environ.get("AS_SERVED_FONT")
    if env and os.path.exists(env):
        return env
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@lru_cache(maxsize=8)
def _load_font(path: str, size_px: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, size_px)


def alpha_warp(canvas: np.ndarray, image_bgra: np.ndarray, dst_quad: np.ndarray,
               opacity: float = 1.0) -> None:
    """BGRA 画像を dst_quad(proj px, TL/TR/BR/BL) へ透視ワープしてアルファ合成する。

    ワープ先の外接矩形だけを処理する。1フレームに十数個の要素を描くので、毎回
    フル解像度(1920x1080)でワープすると投影が数フレーム/秒まで落ちてしまう。
    """
    if opacity <= 0.0:
        return
    proj_h, proj_w = canvas.shape[:2]
    quad = np.asarray(dst_quad, dtype=np.float32)

    x0 = int(np.floor(quad[:, 0].min())) - 1
    y0 = int(np.floor(quad[:, 1].min())) - 1
    x1 = int(np.ceil(quad[:, 0].max())) + 1
    y1 = int(np.ceil(quad[:, 1].max())) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, proj_w), min(y1, proj_h)
    if x1 <= x0 or y1 <= y0:
        return  # 画面外

    src_h, src_w = image_bgra.shape[:2]
    # 画素の「中心」ではなく「外縁」を四隅として対応づける。(w-1) を使うと画像が
    # 1/w だけ大きく描かれ、実寸表示にそのままバイアスとして乗る。
    src = np.array([[-0.5, -0.5], [src_w - 0.5, -0.5],
                    [src_w - 0.5, src_h - 0.5], [-0.5, src_h - 0.5]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, quad - np.array([x0, y0], dtype=np.float32))

    warped = cv2.warpPerspective(image_bgra, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = (warped[:, :, 3:4].astype(np.float32) / 255.0) * float(opacity)
    if not np.any(alpha > 0.0):
        return

    roi = canvas[y0:y1, x0:x1]
    blended = warped[:, :, :3].astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    np.copyto(roi, np.clip(blended, 0, 255).astype(np.uint8))


def draw_image_mm(canvas: np.ndarray, frame: TableFrame, image_bgra: np.ndarray,
                  center_mm: Sequence, width_mm: float, height_mm: Optional[float] = None,
                  angle_deg: float = 0.0, opacity: float = 1.0) -> None:
    """画像を「幅 width_mm の実物」として机の上に置く。height_mm 省略時は元画像の縦横比を保つ。"""
    if height_mm is None:
        h, w = image_bgra.shape[:2]
        height_mm = width_mm * (h / w)
    quad_mm = rect_mm(float(center_mm[0]), float(center_mm[1]), width_mm, height_mm, angle_deg)
    alpha_warp(canvas, image_bgra, frame.table_to_proj(quad_mm), opacity)


def draw_polyline_mm(canvas: np.ndarray, frame: TableFrame, pts_mm: Sequence,
                     color=WHITE, thickness_px: int = 2, closed: bool = True) -> None:
    pts = frame.table_to_proj(pts_mm).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], closed, color, thickness_px, cv2.LINE_AA)


def fill_poly_mm(canvas: np.ndarray, frame: TableFrame, pts_mm: Sequence, color) -> None:
    pts = frame.table_to_proj(pts_mm).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA)


def draw_plate_mm(canvas: np.ndarray, frame: TableFrame, center_mm: Sequence,
                  diameter_mm: float, color=(90, 90, 90), thickness_px: int = 2,
                  dashes: int = 0) -> None:
    """皿の輪郭（実寸の円）。dashes>0 で破線にする（『まだ置かれていない皿』の表現）。"""
    ring = circle_mm(float(center_mm[0]), float(center_mm[1]), diameter_mm, segments=192)
    if dashes <= 0:
        draw_polyline_mm(canvas, frame, ring, color, thickness_px, closed=True)
        return
    n = len(ring)
    seg = max(2, n // (dashes * 2))
    for start in range(0, n, seg * 2):
        chunk = ring[start:start + seg]
        if len(chunk) >= 2:
            draw_polyline_mm(canvas, frame, chunk, color, thickness_px, closed=False)


@lru_cache(maxsize=256)
def _text_image_cached(text: str, height_px: int, color: tuple, font_path: Optional[str],
                       padding_px: int) -> np.ndarray:
    return _render_text_image(text, height_px, color, font_path, padding_px)


def text_image(text: str, height_px: int, color=WHITE, font_path: Optional[str] = None,
               padding_px: int = 4) -> np.ndarray:
    """文字を BGRA のタイト画像にして返す。

    毎フレーム同じ文字列を描き直すので結果をキャッシュする。返る配列は共有なので
    呼び出し側で書き換えないこと（`alpha_warp` は読むだけ）。
    """
    return _text_image_cached(text, int(height_px), tuple(color),
                              font_path or find_font(), int(padding_px))


def _render_text_image(text: str, height_px: int, color, font_path: Optional[str],
                       padding_px: int) -> np.ndarray:
    """フォントが無い環境では OpenCV の内蔵フォントへフォールバックする。"""
    if font_path is None:
        scale = height_px / 22.0
        thickness = max(1, int(round(scale * 1.6)))
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        img = np.zeros((h + baseline + padding_px * 2, w + padding_px * 2, 4), dtype=np.uint8)
        cv2.putText(img, text, (padding_px, h + padding_px), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (color[0], color[1], color[2], 255), thickness, cv2.LINE_AA)
        return img

    from PIL import Image, ImageDraw

    font = _load_font(font_path, int(height_px))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    w = max(1, right - left) + padding_px * 2
    h = max(1, bottom - top) + padding_px * 2

    pil = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(pil).text((padding_px - left, padding_px - top), text, font=font,
                             fill=(color[2], color[1], color[0], 255))  # PIL は RGBA
    rgba = np.array(pil)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def draw_text_mm(canvas: np.ndarray, frame: TableFrame, text: str, anchor_mm: Sequence,
                 height_mm: float, color=WHITE, align: str = "left",
                 valign: str = "top", angle_deg: float = 0.0, opacity: float = 1.0,
                 render_px: int = 96, max_width_mm: Optional[float] = None) -> float:
    """机の上に高さ height_mm の文字を置く。戻り値は描いた文字の幅(mm)。

    align: left / center / right（anchor_mm の x をどう解釈するか）
    valign: top / middle / bottom
    max_width_mm: 指定すると、その幅に収まるよう文字高さを自動的に縮める。
        投影範囲は設置によって変わるので、長い文言が机の外へはみ出すのを防ぐ。
    """
    if not text:
        return 0.0
    img = text_image(text, render_px, color)
    h_px, w_px = img.shape[:2]
    width_mm = height_mm * (w_px / h_px)

    if max_width_mm is not None and width_mm > max_width_mm > 0:
        height_mm *= max_width_mm / width_mm
        width_mm = max_width_mm

    ax, ay = float(anchor_mm[0]), float(anchor_mm[1])
    cx = ax + {"left": width_mm * 0.5, "center": 0.0, "right": -width_mm * 0.5}[align]
    cy = ay + {"top": height_mm * 0.5, "middle": 0.0, "bottom": -height_mm * 0.5}[valign]

    quad_mm = rect_mm(cx, cy, width_mm, height_mm, angle_deg)
    alpha_warp(canvas, img, frame.table_to_proj(quad_mm), opacity)
    return width_mm


def draw_panel_mm(canvas: np.ndarray, frame: TableFrame, center_mm: Sequence,
                  width_mm: float, height_mm: float, color=(28, 28, 28),
                  border=(90, 90, 90), angle_deg: float = 0.0) -> None:
    """情報パネルの下地。投影なので『黒＝光を出さない』であり、暗い色ほど机の地色が出る。"""
    quad = rect_mm(float(center_mm[0]), float(center_mm[1]), width_mm, height_mm, angle_deg)
    fill_poly_mm(canvas, frame, quad, color)
    if border is not None:
        draw_polyline_mm(canvas, frame, quad, border, 2, closed=True)
