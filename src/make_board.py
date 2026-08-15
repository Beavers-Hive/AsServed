"""
make_board.py
寸法較正ボード（ArUco マーカーを既知の mm 間隔で並べた印刷用シート）を生成する。
カメラ・プロジェクター不要。

このボードが「実寸」の基準になる。ボードを机に置いて calibrate_metric.py を回すと、
テーブル平面の mm 座標 → プロジェクター px のホモグラフィが求まり、以後アプリは
すべて mm で描けるようになる。

出力:
  markers/board_<dict>_<marker>mm_<pitch>mm.png   印刷用 PNG（指定 DPI の実寸）
  markers/board_layout.json                        各マーカー4隅の mm 座標（公称値）

使い方:
  uv run python src/make_board.py                  # 既定: 60mm マーカー / 90mm ピッチ / 3x2
  uv run python src/make_board.py --marker-mm 50 --pitch-mm 80 --cols 3 --rows 2

印刷時の注意（ここを外すと実寸が全部ずれる）:
  1. プリンタ設定を「実際のサイズ / 100% / 拡大縮小なし」にする
  2. 印刷後、必ず定規で「マーカー1辺（黒枠込み）」と「隣り合うマーカーの左端どうしの
     間隔（ピッチ）」を実測する
  3. 公称値とずれていたら calibrate_metric.py に --measured-marker-mm で実測値を渡す
     （レイアウト全体が相似にスケールされる）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_marker import resolve_dictionary  # noqa: E402


PAPERS = {                # 用紙の実寸(mm)。短辺 x 長辺
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
    "letter": (215.9, 279.4),
    "b5": (176.0, 250.0),
}
PRINT_MARGIN_MM = 5.0     # 家庭用プリンタが確保する余白の目安（全周）


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def save_png_with_dpi(img: np.ndarray, path: str, dpi: int) -> None:
    """PNG に解像度情報(pHYs)を埋めて保存する。

    cv2.imwrite は DPI を書かない。DPI の無い PNG を印刷しようとすると、
    macOS のプレビューなどは 72dpi と解釈するため、300dpi 前提の画像が
    約4.2倍に拡大されてしまう（「100%にしたら巨大になった」の原因）。
    """
    from PIL import Image

    Image.fromarray(img).save(path, dpi=(dpi, dpi))


def compose_print_page(board: np.ndarray, board_w_mm: float, board_h_mm: float,
                       paper: str, dpi: int) -> tuple:
    """用紙1枚にボードを実寸で配置したページ画像を作る。戻り値は (画像, 向き, ページmm)。

    PDF にすればページの物理サイズが絶対値として入るので、「実際のサイズで印刷」が
    アプリ側の DPI 解釈に左右されなくなる。あわせて印刷スケールをその場で検算できる
    100mm の物差しを余白に刷り込む。
    """
    short, long_ = PAPERS[paper]
    if board_w_mm <= long_ - PRINT_MARGIN_MM * 2 and board_h_mm <= short - PRINT_MARGIN_MM * 2:
        page_w_mm, page_h_mm, orientation = long_, short, "landscape"
    else:
        page_w_mm, page_h_mm, orientation = short, long_, "portrait"

    page = np.full((mm_to_px(page_h_mm, dpi), mm_to_px(page_w_mm, dpi)), 255, dtype=np.uint8)
    bh, bw = board.shape[:2]

    # 物差しをボードの下に置きたいので、その分だけボードを上に寄せる
    ruler_band_mm = 22.0
    room_below = page_h_mm - board_h_mm - PRINT_MARGIN_MM * 2
    with_ruler = room_below >= ruler_band_mm

    x0 = (page.shape[1] - bw) // 2
    if with_ruler:
        y0 = mm_to_px((page_h_mm - board_h_mm - ruler_band_mm) * 0.5, dpi)
    else:
        y0 = (page.shape[0] - bh) // 2
    page[y0:y0 + bh, x0:x0 + bw] = board

    if with_ruler:
        ry = y0 + bh + mm_to_px(10.0, dpi)
        rx = x0
        length = mm_to_px(100.0, dpi)
        tick = mm_to_px(3.0, dpi)
        cv2.line(page, (rx, ry), (rx + length, ry), 0, max(1, dpi // 150))
        for i in range(11):
            x = rx + int(round(length * i / 10))
            h = tick * (2 if i % 5 == 0 else 1)
            cv2.line(page, (x, ry - h), (x, ry), 0, max(1, dpi // 200))
        cv2.putText(page, "100 mm  <- measure this. if it is not 100.0 mm, the print was scaled.",
                    (rx, ry + mm_to_px(6.0, dpi)), cv2.FONT_HERSHEY_SIMPLEX,
                    dpi / 300.0 * 0.7, 0, max(1, dpi // 300), cv2.LINE_AA)

    return page, orientation, (page_w_mm, page_h_mm)


def check_paper_fit(board_w_mm: float, board_h_mm: float, paper: str) -> None:
    """指定用紙に実寸で載るかを判定し、向きを含めて案内する。

    「用紙に合わせる」で勝手に縮小されるのが寸法較正の一番危ない落とし穴なので、
    生成時点で必要な向きを言い切っておく。
    """
    if paper not in PAPERS:
        return
    short, long_ = PAPERS[paper]
    usable_short = short - PRINT_MARGIN_MM * 2
    usable_long = long_ - PRINT_MARGIN_MM * 2
    name = paper.upper()

    portrait = board_w_mm <= usable_short and board_h_mm <= usable_long
    landscape = board_w_mm <= usable_long and board_h_mm <= usable_short

    if landscape and not portrait:
        print(f"[make_board] 用紙: {name} 横向き（landscape）で印刷してください。")
    elif portrait and not landscape:
        print(f"[make_board] 用紙: {name} 縦向き（portrait）で印刷してください。")
    elif portrait and landscape:
        print(f"[make_board] 用紙: {name} はどちらの向きでも入ります。")
    else:
        print(f"[make_board] 警告: ボード {board_w_mm:g} x {board_h_mm:g} mm は "
              f"{name}（余白 {PRINT_MARGIN_MM:g}mm 込みで "
              f"{usable_short:g} x {usable_long:g} mm）に実寸で入りません。")
        print(f"[make_board]   --marker-mm / --pitch-mm / --cols / --rows を小さくするか、")
        print(f"[make_board]   --paper a3 など大きい用紙を使ってください。")
        print(f"[make_board]   {name} 縦向きに収める例:")
        print(f"[make_board]     uv run python src/make_board.py --cols 2 --rows 3 "
              f"--marker-mm 50 --pitch-mm 75")
        return

    print(f"[make_board] 印刷設定は必ず『実際のサイズ / 100% / 用紙に合わせない』にしてください。")
    print(f"[make_board] 『用紙に合わせる』が有効だと数%縮み、その分そのまま投影がずれます。")


def build_board(dict_name: str, ids: list[int], marker_mm: float, pitch_mm: float,
                cols: int, rows: int, dpi: int, margin_mm: float):
    """ボード画像と、各マーカー4隅の mm 座標（ボード原点=左上マージン端）を返す。

    mm 座標系は画像と同じ向き（x 右、y 下）にしておく。カメラで真上気味に見たときの
    見え方と一致するので、デバッグ時に頭を使わずに済む。
    """
    need = cols * rows
    if len(ids) < need:
        raise ValueError(f"ID が足りません: {cols}x{rows}={need} 必要、指定は {len(ids)} 個")
    ids = ids[:need]

    board_w_mm = margin_mm * 2 + pitch_mm * (cols - 1) + marker_mm
    board_h_mm = margin_mm * 2 + pitch_mm * (rows - 1) + marker_mm
    img = np.full((mm_to_px(board_h_mm, dpi), mm_to_px(board_w_mm, dpi)), 255, dtype=np.uint8)

    layout: dict[str, list[list[float]]] = {}
    side_px = mm_to_px(marker_mm, dpi)
    aruco_dict, _ = resolve_dictionary(dict_name)

    for k, marker_id in enumerate(ids):
        r, c = divmod(k, cols)
        x_mm = margin_mm + pitch_mm * c
        y_mm = margin_mm + pitch_mm * r
        x0, y0 = mm_to_px(x_mm, dpi), mm_to_px(y_mm, dpi)

        # 白余白をつけずマーカー本体だけを正確な px で貼る（余白を挟むと実寸がずれる）
        tile = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px)
        img[y0:y0 + side_px, x0:x0 + side_px] = tile

        # cv2.aruco の返す順序 TL,TR,BR,BL に合わせる
        layout[str(marker_id)] = [
            [x_mm, y_mm],
            [x_mm + marker_mm, y_mm],
            [x_mm + marker_mm, y_mm + marker_mm],
            [x_mm, y_mm + marker_mm],
        ]

        label = f"id{marker_id}"
        cv2.putText(img, label, (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    dpi / 300.0 * 0.6, 0, max(1, dpi // 300), cv2.LINE_AA)

    meta = {
        "dict": dict_name,
        "marker_mm": marker_mm,
        "pitch_mm": pitch_mm,
        "cols": cols,
        "rows": rows,
        "dpi": dpi,
        "margin_mm": margin_mm,
        "board_size_mm": [board_w_mm, board_h_mm],
        "corners_mm": layout,
        "note": "corners_mm は TL,TR,BR,BL 順。印刷実測とずれる場合は calibrate_metric.py の --measured-marker-mm で補正する。",
    }
    return img, meta


def main() -> int:
    p = argparse.ArgumentParser(description="寸法較正ボード（既知 mm 間隔の ArUco シート）を生成")
    p.add_argument("--dict", default="DICT_4X4_50")
    p.add_argument("--ids", default="10,11,12,13,14,15", help="使う ID をカンマ区切りで")
    p.add_argument("--marker-mm", type=float, default=60.0)
    p.add_argument("--pitch-mm", type=float, default=90.0, help="隣接マーカーの左端どうしの間隔")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--margin-mm", type=float, default=10.0)
    p.add_argument("--paper", default="a4", choices=list(PAPERS) + ["none"],
                   help="印刷する用紙。実寸で載るかを判定して向きを案内する（既定 a4）")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "markers"))
    args = p.parse_args()

    if args.pitch_mm <= args.marker_mm:
        raise SystemExit(
            f"--pitch-mm({args.pitch_mm}) は --marker-mm({args.marker_mm}) より大きくしてください（マーカーが重なります）"
        )

    ids = [int(s) for s in args.ids.split(",") if s.strip()]
    img, meta = build_board(args.dict, ids, args.marker_mm, args.pitch_mm,
                            args.cols, args.rows, args.dpi, args.margin_mm)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    stem = f"board_{args.dict}_{args.marker_mm:g}mm_{args.pitch_mm:g}mm"
    png = os.path.join(outdir, stem + ".png")
    js = os.path.join(outdir, "board_layout.json")
    save_png_with_dpi(img, png, args.dpi)
    with open(js, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    w_mm, h_mm = meta["board_size_mm"]
    print(f"[make_board] {png}")
    print(f"[make_board] {js}")
    print(f"[make_board] 用紙上の実寸 {w_mm:g} x {h_mm:g} mm / {args.dpi}dpi "
          f"({img.shape[1]}x{img.shape[0]} px)")

    # 用紙に載るなら PDF も出す。PDF はページの物理サイズを絶対値で持つので、
    # 「実際のサイズで印刷」がアプリの DPI 解釈に左右されない。印刷はこちらを推奨。
    fits = args.paper in PAPERS and (
        (w_mm <= PAPERS[args.paper][1] - PRINT_MARGIN_MM * 2
         and h_mm <= PAPERS[args.paper][0] - PRINT_MARGIN_MM * 2)
        or (w_mm <= PAPERS[args.paper][0] - PRINT_MARGIN_MM * 2
            and h_mm <= PAPERS[args.paper][1] - PRINT_MARGIN_MM * 2))
    if fits:
        page, orientation, (pw, ph) = compose_print_page(img, w_mm, h_mm, args.paper, args.dpi)
        pdf = os.path.join(outdir, f"{stem}_{args.paper}_{orientation}.pdf")
        from PIL import Image

        Image.fromarray(page).save(pdf, resolution=float(args.dpi))
        print(f"[make_board] {pdf}")
        print(f"[make_board] ★ 印刷はこの PDF を使ってください（ページ = {args.paper.upper()} "
              f"{orientation} {pw:g} x {ph:g} mm、ボードは実寸で配置済み）")
        print(f"[make_board]   ページ下部の 100mm 物差しを定規で測れば、印刷スケールをその場で検算できます。")

    check_paper_fit(w_mm, h_mm, args.paper)
    print(f"[make_board] 印刷後、必ずマーカー1辺（黒枠込み）を定規で実測してください。")
    print(f"[make_board]   公称 {args.marker_mm:g}mm とずれていたら:")
    print(f"[make_board]   uv run python src/calibrate_metric.py --measured-marker-mm <実測値>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
