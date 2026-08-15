"""
make_marker.py
印刷用の ArUco マーカー PNG を生成する（カメラ不要）。

生成したマーカーを印刷して箱の上面に貼り、そのマーカーを RGB カメラで検出して
箱の位置・向きを追跡する。

OpenCV 4.11 の新 API（cv2.aruco.getPredefinedDictionary / generateImageMarker）を使う。

実行例:
  uv run python src/make_marker.py                       # 既定: DICT_4X4_50, id=0, 600px, 白余白つき
  uv run python src/make_marker.py --id 3 --size 800
  uv run python src/make_marker.py --dict DICT_5X5_100 --id 0
出力:
  markers/aruco_<dict>_id<ID>_<size>px.png
"""
from __future__ import annotations

import argparse
import os
import sys


def resolve_dictionary(name: str):
    """"DICT_4X4_50" のような名前から cv2.aruco の定義済み辞書を取得する。"""
    import cv2

    const = getattr(cv2.aruco, name, None)
    if const is None:
        raise ValueError(
            f"未知の ArUco 辞書名: {name!r}. 例: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250 など。"
        )
    return cv2.aruco.getPredefinedDictionary(const), const


def make_marker_image(dict_name: str, marker_id: int, size_px: int, border_ratio: float):
    """マーカー画像（白余白つき, グレースケール uint8）を生成して返す。"""
    import cv2
    import numpy as np

    aruco_dict, _ = resolve_dictionary(dict_name)
    # 新API: generateImageMarker（旧 drawMarker の後継）
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)

    border = max(1, int(size_px * border_ratio))
    canvas = np.full((size_px + 2 * border, size_px + 2 * border), 255, dtype=np.uint8)
    canvas[border:border + size_px, border:border + size_px] = marker
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="印刷用 ArUco マーカー PNG を生成")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco 辞書名（既定 DICT_4X4_50）")
    parser.add_argument("--id", type=int, default=0, help="マーカー ID（既定 0）")
    parser.add_argument("--size", type=int, default=600, help="マーカー本体の一辺 px（既定 600）")
    parser.add_argument("--border-ratio", type=float, default=0.15,
                        help="マーカー辺に対する白余白の割合（既定 0.15）")
    parser.add_argument("--out-dir", default=None, help="出力ディレクトリ（既定: プロジェクト直下 markers/）")
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except Exception as exc:
        print(f"[make_marker] OpenCV をインポートできません: {exc!r}")
        return 1

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(project_root, "markers")
    os.makedirs(out_dir, exist_ok=True)

    try:
        img = make_marker_image(args.dict, args.id, args.size, args.border_ratio)
    except Exception as exc:
        print(f"[make_marker] マーカー生成に失敗: {exc!r}")
        return 1

    import cv2
    fname = f"aruco_{args.dict}_id{args.id}_{args.size}px.png"
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, img)
    print(f"[make_marker] 生成しました: {path}  (画像サイズ {img.shape[1]}x{img.shape[0]}px)")
    print("[make_marker] これを印刷し、実測した『黒枠を含むマーカー1辺の長さ(メートル)』を")
    print("[make_marker]   config.json の aruco.marker_length_m に設定してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
