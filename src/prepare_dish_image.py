"""
prepare_dish_image.py
料理写真を「実寸投影用の素材」に整える。白背景を透過にし、料理の外接矩形で切り詰め、
正方形に整えて保存する。

このアプリは **画像の幅 = menu.json の food_diameter_mm** として投影する。つまり
料理のまわりに余白が残っていると、その分だけ料理が小さく投影される。目視では
気づきにくく、実寸の主張が静かに崩れるので、切り詰めは必ずこのスクリプトで行う。

実行:
    uv run python src/prepare_dish_image.py assets/dishes/_raw/margherita.png \
        -o assets/dishes/pizza_margherita.png

    # 背景が白でない/抜けが甘いとき
    uv run python src/prepare_dish_image.py in.png -o out.png --threshold 18 --feather 3

    # 抜き結果の確認用（アルファをチェッカーボードに重ねた画像も出す）
    uv run python src/prepare_dish_image.py in.png -o out.png --contact-sheet check.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def background_mask(bgr: np.ndarray, threshold: int, corner_sample: int = 24) -> np.ndarray:
    """背景（四隅の色に近い領域）を 0、被写体を 255 とするマスクを返す。

    背景色は四隅から推定する。白決め打ちにすると、生成画像がわずかにグレーがかって
    いた場合に何も抜けない。
    """
    h, w = bgr.shape[:2]
    s = corner_sample
    corners = np.concatenate([
        bgr[:s, :s].reshape(-1, 3), bgr[:s, -s:].reshape(-1, 3),
        bgr[-s:, :s].reshape(-1, 3), bgr[-s:, -s:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(corners, axis=0)

    dist = np.linalg.norm(bgr.astype(np.float32) - bg[None, None, :], axis=2)
    mask = (dist > float(threshold)).astype(np.uint8) * 255

    # 小さな穴とゴミを埋める。バジルの葉の隙間などを拾いすぎないよう控えめに。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    """最大の連結成分だけを残す。背景のノイズや落ち影を切り落とすため。"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num <= 1:
        raise SystemExit("[prepare_dish_image] 被写体を検出できませんでした。--threshold を下げてください。")
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == biggest] = 255
    # 内側の穴（トッピングの隙間から背景色が覗く箇所）を塗りつぶす
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def prepare(bgr: np.ndarray, threshold: int, feather: int, size: int) -> tuple:
    mask = largest_component(background_mask(bgr, threshold))

    xs = np.where(mask.max(axis=0) > 0)[0]
    ys = np.where(mask.max(axis=1) > 0)[0]
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    crop_bgr, crop_mask = bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    ch, cw = crop_mask.shape[:2]

    # 正方形に詰める。料理は丸いので外接矩形はほぼ正方形になるが、わずかな差を
    # 中央寄せで吸収する。ここで比率を変えると実寸がゆがむので、拡縮ではなく余白で合わせる。
    side = max(cw, ch)
    square_bgr = np.zeros((side, side, 3), dtype=np.uint8)
    square_mask = np.zeros((side, side), dtype=np.uint8)
    ox, oy = (side - cw) // 2, (side - ch) // 2
    square_bgr[oy:oy + ch, ox:ox + cw] = crop_bgr
    square_mask[oy:oy + ch, ox:ox + cw] = crop_mask

    if feather > 0:
        k = feather * 2 + 1
        square_mask = cv2.GaussianBlur(square_mask, (k, k), 0)

    bgra = np.dstack([square_bgr, square_mask])
    if size > 0 and side != size:
        bgra = cv2.resize(bgra, (size, size), interpolation=cv2.INTER_AREA)

    info = {
        "source_size": (bgr.shape[1], bgr.shape[0]),
        "bbox": (x0, y0, x1, y1),
        "bbox_size": (cw, ch),
        "aspect": cw / ch,
        "output_size": bgra.shape[1],
    }
    return bgra, info


def contact_sheet(bgra: np.ndarray, tile: int = 32) -> np.ndarray:
    """アルファの抜け具合を目視するため、チェッカーボードに重ねた画像を作る。"""
    h, w = bgra.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // tile) + (xx // tile)) % 2 == 0, 200, 120).astype(np.uint8)
    board = np.dstack([board] * 3)
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(bgra[:, :, :3].astype(np.float32) * a
                   + board.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser(description="料理写真を実寸投影用の透過素材に整える")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--threshold", type=int, default=24,
                   help="背景とみなす色距離のしきい値（大きいほど抜けにくい）")
    p.add_argument("--feather", type=int, default=2, help="輪郭のぼかし量(px)")
    p.add_argument("--size", type=int, default=1024, help="出力の一辺(px)。0 で原寸のまま")
    p.add_argument("--contact-sheet", default=None, help="抜き確認用の合成画像も書き出す")
    args = p.parse_args()

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"[prepare_dish_image] 読み込めません: {args.input}")

    bgra, info = prepare(bgr, args.threshold, args.feather, args.size)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), bgra)

    cw, ch = info["bbox_size"]
    print(f"[prepare_dish_image] {args.input} -> {out}")
    print(f"[prepare_dish_image] 元画像 {info['source_size'][0]}x{info['source_size'][1]} / "
          f"料理の外接矩形 {cw}x{ch} px (縦横比 {info['aspect']:.3f})")
    print(f"[prepare_dish_image] 出力 {info['output_size']}x{info['output_size']} px（余白なし）")
    if abs(info["aspect"] - 1.0) > 0.06:
        print(f"[prepare_dish_image] 警告: 縦横比が 1.0 から離れています。真上から撮れていないか、"
              f"背景の抜けが不完全な可能性があります。")
    print(f"[prepare_dish_image] menu.json の food_diameter_mm には、"
          f"この料理の実際の直径(mm)を入れてください。")

    if args.contact_sheet:
        cv2.imwrite(args.contact_sheet, contact_sheet(bgra))
        print(f"[prepare_dish_image] 抜き確認用: {args.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
