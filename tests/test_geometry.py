"""geometry.py の変換が実寸として正しいかを機材なしで確かめる。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from geometry import (  # noqa: E402
    TableFrame,
    circle_mm,
    rect_mm,
    solve_table_to_proj,
    transform_points,
)


def make_homography(scale_px_per_mm: float = 3.0, tilt: float = 0.0005) -> np.ndarray:
    """机を斜めから投影したときのような遠近つきの mm→px 行列を作る。"""
    return np.array([
        [scale_px_per_mm, 0.0, 200.0],
        [0.0, scale_px_per_mm, 100.0],
        [0.0, tilt, 1.0],
    ], dtype=np.float64)


def test_rect_mm_has_requested_physical_size():
    quad = rect_mm(100.0, 50.0, 200.0, 80.0)
    assert quad.shape == (4, 2)
    assert np.isclose(np.linalg.norm(quad[1] - quad[0]), 200.0)   # 上辺 = 幅
    assert np.isclose(np.linalg.norm(quad[3] - quad[0]), 80.0)    # 左辺 = 高さ
    assert np.allclose(quad.mean(axis=0), [100.0, 50.0])


def test_rect_mm_rotation_preserves_size():
    quad = rect_mm(0.0, 0.0, 120.0, 60.0, angle_deg=37.0)
    assert np.isclose(np.linalg.norm(quad[1] - quad[0]), 120.0)
    assert np.isclose(np.linalg.norm(quad[3] - quad[0]), 60.0)


def test_circle_mm_diameter():
    ring = circle_mm(10.0, -20.0, 220.0, segments=64)
    r = np.linalg.norm(ring - np.array([10.0, -20.0]), axis=1)
    assert np.allclose(r, 110.0)


def test_solve_recovers_known_homography():
    H_true = make_homography()
    pts_mm = np.array([[0, 0], [300, 0], [300, 200], [0, 200],
                       [90, 90], [210, 40]], dtype=np.float32)
    pts_proj = transform_points(H_true, pts_mm)

    H, rms = solve_table_to_proj(pts_mm, pts_proj)
    assert rms < 1e-3, f"残差が大きすぎます: {rms}"

    # ホモグラフィはスケール不定なので、行列比較ではなく点の一致で見る
    assert np.allclose(transform_points(H, pts_mm), pts_proj, atol=1e-2)


def test_solve_reports_residual_in_mm():
    """proj 空間ではなく mm へ引き戻した誤差を返しているか。"""
    H_true = make_homography(scale_px_per_mm=4.0, tilt=0.0)
    # 4点ちょうどだと必ず厳密解になり残差が出ないので、余剰点を入れて最小二乗にする
    pts_mm = np.array([[0, 0], [300, 0], [300, 200], [0, 200],
                       [150, 0], [150, 200]], dtype=np.float32)
    pts_proj = transform_points(H_true, pts_mm)

    # 1点だけ proj 上で 4px ずらす。4px/mm なので mm 換算では約 1mm のずれ
    noisy = pts_proj.copy()
    noisy[0, 0] += 4.0
    _H, rms = solve_table_to_proj(pts_mm, noisy)
    assert 0.1 < rms < 1.5, f"mm 単位の残差になっていません: {rms}"


def test_solve_rejects_too_few_points():
    with pytest.raises(ValueError):
        solve_table_to_proj(np.zeros((3, 2)), np.zeros((3, 2)))


def test_table_frame_round_trip_and_cam_chain():
    H_tp = make_homography()
    H_cp = np.array([[2.0, 0.1, -30.0], [0.0, 2.2, 15.0], [0.0, 0.0, 1.0]])
    frame = TableFrame(H_table_to_proj=H_tp, H_cam_to_proj=H_cp)

    pts_mm = np.array([[0.0, 0.0], [150.0, 220.0]], dtype=np.float32)
    back = frame.proj_to_table(frame.table_to_proj(pts_mm))
    assert np.allclose(back, pts_mm, atol=1e-3)

    # cam → table は cam → proj → table と一致するはず
    cam_pts = np.array([[100.0, 80.0], [640.0, 360.0]], dtype=np.float32)
    via_proj = frame.proj_to_table(transform_points(H_cp, cam_pts))
    assert np.allclose(frame.cam_to_table(cam_pts), via_proj, atol=1e-3)


def test_cam_to_table_requires_calibration():
    frame = TableFrame(H_table_to_proj=make_homography())
    with pytest.raises(ValueError):
        _ = frame.H_cam_to_table


def _point_in_quad(p, quad) -> bool:
    """凸四角形の内側か（外積の符号が全辺で揃うか）。"""
    signs = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        signs.append(np.sign(np.cross(b - a, np.asarray(p) - a)))
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def test_inscribed_bounds_stay_inside_the_projected_quad():
    """内接矩形の四隅が、実際に投影される台形の内側にあること。

    外接矩形を使うと台形の外へ UI がはみ出し、実機で「上端の文字が切れる」。
    """
    frame = TableFrame(H_table_to_proj=np.array([
        [2.6, 0.10, 520.0],
        [-0.12, 2.55, 620.0],
        [-0.00004, 0.00025, 1.0],
    ]))
    quad, inner = frame.projection_bounds_mm(1920, 1080, inscribed=True)
    _quad, outer = frame.projection_bounds_mm(1920, 1080, inscribed=False)

    for corner in ((inner[0], inner[1]), (inner[2], inner[1]),
                   (inner[2], inner[3]), (inner[0], inner[3])):
        assert _point_in_quad(corner, quad), f"内接矩形の角 {corner} が台形の外です"

    assert inner[0] >= outer[0] and inner[1] >= outer[1]
    assert inner[2] <= outer[2] and inner[3] <= outer[3]
    assert (inner[2] - inner[0]) * (inner[3] - inner[1]) < \
           (outer[2] - outer[0]) * (outer[3] - outer[1])


def test_inscribed_equals_outer_when_there_is_no_perspective():
    frame = TableFrame(H_table_to_proj=np.array([[3.0, 0.0, 100.0],
                                                 [0.0, 3.0, 50.0],
                                                 [0.0, 0.0, 1.0]]))
    _q, inner = frame.projection_bounds_mm(1920, 1080, inscribed=True)
    _q, outer = frame.projection_bounds_mm(1920, 1080, inscribed=False)
    assert np.allclose(inner, outer)


def test_px_per_mm_matches_pure_scale():
    frame = TableFrame(H_table_to_proj=make_homography(scale_px_per_mm=3.0, tilt=0.0))
    assert np.isclose(frame.px_per_mm_at(120.0, 90.0), 3.0, atol=1e-6)
