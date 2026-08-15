"""寸法較正のパイプラインを、カメラもプロジェクターも使わず合成画像で通しで検証する。

やっていること:
  1. make_board.py で生成するのと同じボード画像を作る（mm ↔ 画素の対応は既知）
  2. 「机を斜めから撮ったカメラ画像」を既知のホモグラフィで合成する
  3. ArucoDetector で検出し、calibrate_metric.py と同じ手順で H_table_mm→proj を解く
  4. 解いた変換で 100mm を測り直し、本当に 100mm に戻るかを見る

ここが通っていれば、実機でズレたときに「アルゴリズムではなく較正手順か機材の問題」と
切り分けられる。マーカー4隅の順序(TL,TR,BR,BL)を取り違える類のバグはここで落ちる。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector_aruco import ArucoDetector  # noqa: E402
from geometry import solve_table_to_proj, transform_points  # noqa: E402
from make_board import build_board  # noqa: E402

DICT = "DICT_4X4_50"
IDS = [10, 11, 12, 13, 14, 15]
MARKER_MM = 60.0
PITCH_MM = 90.0
BOARD_DPI = 150  # テストを速くするため印刷用より粗くする


def synth_camera_view(board_img: np.ndarray, H_board_px_to_cam: np.ndarray,
                      size=(1280, 720)) -> np.ndarray:
    """ボード画像を「カメラから見た机」として合成する（周囲は机の灰色）。"""
    bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    canvas = np.full((size[1], size[0], 3), 150, dtype=np.uint8)
    warped = cv2.warpPerspective(bgr, H_board_px_to_cam, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_TRANSPARENT, dst=canvas)
    return warped


def test_metric_calibration_recovers_true_scale():
    board_img, meta = build_board(DICT, IDS, MARKER_MM, PITCH_MM, cols=3, rows=2,
                                  dpi=BOARD_DPI, margin_mm=10.0)
    px_per_mm_board = BOARD_DPI / 25.4

    # 机を斜め上から見たカメラを模す（遠近つき）
    H_board_px_to_cam = np.array([
        [0.62, 0.05, 190.0],
        [0.02, 0.60, 90.0],
        [0.00006, 0.00010, 1.0],
    ], dtype=np.float64)
    cam_img = synth_camera_view(board_img, H_board_px_to_cam)

    # プロジェクター側も既知の変換にしておく（実機では calibrate.py が作る H_cam→proj）
    H_cam_to_proj = np.array([
        [1.45, 0.08, -120.0],
        [-0.05, 1.50, -60.0],
        [0.00002, 0.00007, 1.0],
    ], dtype=np.float64)

    detector = ArucoDetector({"aruco": {"dict": DICT, "id": IDS[0], "ema_alpha": 1.0}})
    found = detector.detect_all(cam_img, set(IDS))
    assert len(found) == len(IDS), f"検出できたのは {sorted(found)} だけです"

    pts_mm, pts_proj = [], []
    for marker_id, quad in sorted(found.items()):
        pts_mm.append(np.array(meta["corners_mm"][str(marker_id)], dtype=np.float64))
        pts_proj.append(transform_points(H_cam_to_proj, quad.corners))
    pts_mm = np.concatenate(pts_mm)
    pts_proj = np.concatenate(pts_proj)

    H_table_to_proj, rms_mm = solve_table_to_proj(pts_mm, pts_proj)
    assert rms_mm < 1.0, f"残差が大きすぎます: {rms_mm:.2f} mm"

    # 真値: board 画素 → cam → proj を通した変換と一致するはず
    H_mm_to_board_px = np.diag([px_per_mm_board, px_per_mm_board, 1.0])
    H_true = H_cam_to_proj @ H_board_px_to_cam @ H_mm_to_board_px

    probe_mm = np.array([[0, 0], [200, 0], [200, 140], [0, 140], [95, 75]], dtype=np.float32)
    assert np.allclose(transform_points(H_table_to_proj, probe_mm),
                       transform_points(H_true, probe_mm), atol=2.0)

    # 100mm と指定したものが 100mm として戻ってくるか（実寸の最終確認）
    H_proj_to_table = np.linalg.inv(H_table_to_proj)
    for anchor in ([20.0, 20.0], [150.0, 120.0], [230.0, 40.0]):
        seg = np.array([anchor, [anchor[0] + 100.0, anchor[1]]], dtype=np.float32)
        back = transform_points(H_proj_to_table, transform_points(H_table_to_proj, seg))
        measured = float(np.linalg.norm(back[1] - back[0]))
        assert abs(measured - 100.0) < 0.5, f"{anchor} で {measured:.2f} mm になりました"


def test_detector_returns_corners_in_tl_tr_br_bl_order():
    """マーカーを傾けずに写したとき、4隅が TL,TR,BR,BL の順で返ること。

    この順序が崩れると、寸法較正は数値的には解けてしまうのに投影が鏡像・回転する。
    """
    board_img, meta = build_board(DICT, [10], MARKER_MM, PITCH_MM, cols=1, rows=1,
                                  dpi=BOARD_DPI, margin_mm=10.0)
    bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    canvas = np.full((h + 120, w + 160, 3), 150, dtype=np.uint8)
    canvas[60:60 + h, 80:80 + w] = bgr

    detector = ArucoDetector({"aruco": {"dict": DICT, "id": 10, "ema_alpha": 1.0}})
    quad = detector.detect_all(canvas, {10})[10]
    c = quad.corners

    assert c[0][0] < c[1][0], "TL が TR より右にあります"
    assert c[3][0] < c[2][0], "BL が BR より右にあります"
    assert c[0][1] < c[3][1], "TL が BL より下にあります"
    assert c[1][1] < c[2][1], "TR が BR より下にあります"

    side_px = np.linalg.norm(c[1] - c[0])
    expected = MARKER_MM * BOARD_DPI / 25.4
    assert abs(side_px - expected) < 2.0
