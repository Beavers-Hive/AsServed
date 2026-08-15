"""
geometry.py
座標系の変換をまとめる。本プロジェクトの中心概念は「テーブル平面のミリメートル座標」。

このアプリの売りは「実寸で見せる」ことなので、描画は一貫して mm で行い、
最後にプロジェクターのピクセルへ変換する。そのために3つの座標系を扱う。

  cam   : カメラ画像のピクセル座標
  proj  : プロジェクターのフレームバッファのピクセル座標
  table : テーブル平面のミリメートル座標（原点・向きは較正ボードが決める）

いずれも同一平面（机の天板）上の点なので、相互の変換はすべて 3x3 ホモグラフィ1枚で足りる。

  H_cam_to_proj   : calibration.json     （既存の4隅クリック較正）
  H_table_to_proj : metric.json          （較正ボードによる寸法較正）
  H_cam_to_table  : 上記2つから導出（inv(H_table_to_proj) @ H_cam_to_proj）

注意: mm 座標からプロジェクター px への倍率は場所によって変わる（机を斜めから
投影するため遠近がつく）。だから「1mm = N px」という単一のスカラーは存在せず、
必ずホモグラフィで点ごとに変換する。`px_per_mm_at()` はあくまで局所的な目安であり、
デバッグ表示以外には使わないこと。
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def transform_points(H: np.ndarray, pts: Sequence) -> np.ndarray:
    """(N,2) の点群をホモグラフィ H で変換して (N,2) で返す。"""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H.astype(np.float32)).reshape(-1, 2)


def rect_mm(cx: float, cy: float, width_mm: float, height_mm: float,
            angle_deg: float = 0.0) -> np.ndarray:
    """table(mm) 上の矩形4隅を TL,TR,BR,BL 順で返す。

    angle_deg は反時計回り。画像テクスチャの頂点順（TL,TR,BR,BL）に合わせてある。
    """
    hw, hh = width_mm * 0.5, height_mm * 0.5
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    if angle_deg:
        t = np.radians(angle_deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
        local = local @ R.T
    return (local + np.array([cx, cy], dtype=np.float32)).astype(np.float32)


def circle_mm(cx: float, cy: float, diameter_mm: float, segments: int = 96) -> np.ndarray:
    """table(mm) 上の円周点列を返す。皿の輪郭を描くのに使う。"""
    t = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False, dtype=np.float32)
    r = float(diameter_mm) * 0.5
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class TableFrame:
    """table(mm) ↔ proj(px) ↔ cam(px) の変換をまとめて持つ。"""

    H_table_to_proj: np.ndarray
    H_cam_to_proj: Optional[np.ndarray] = None
    board_rms_mm: Optional[float] = None  # 寸法較正の残差（mm）。品質の目安

    # --- 生成 ---------------------------------------------------------------

    @staticmethod
    def load(metric_path: Path, calibration_path: Optional[Path] = None) -> "TableFrame":
        metric = json.loads(Path(metric_path).read_text(encoding="utf-8"))
        H_tp = np.array(metric["homography_table_mm_to_proj"], dtype=np.float64)

        H_cp = None
        if calibration_path is not None and Path(calibration_path).exists():
            calib = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            H_cp = np.array(calib["homography_cam_to_proj"], dtype=np.float64)

        return TableFrame(
            H_table_to_proj=H_tp,
            H_cam_to_proj=H_cp,
            board_rms_mm=metric.get("residual_rms_mm"),
        )

    # --- 変換 ---------------------------------------------------------------

    @property
    def H_proj_to_table(self) -> np.ndarray:
        return np.linalg.inv(self.H_table_to_proj)

    @property
    def H_cam_to_table(self) -> np.ndarray:
        if self.H_cam_to_proj is None:
            raise ValueError(
                "calibration.json（H_cam→proj）が読み込まれていないため cam→table は計算できません。"
            )
        return self.H_proj_to_table @ self.H_cam_to_proj

    def table_to_proj(self, pts_mm: Sequence) -> np.ndarray:
        return transform_points(self.H_table_to_proj, pts_mm)

    def proj_to_table(self, pts_px: Sequence) -> np.ndarray:
        return transform_points(self.H_proj_to_table, pts_px)

    def cam_to_table(self, pts_px: Sequence) -> np.ndarray:
        return transform_points(self.H_cam_to_table, pts_px)

    # --- 診断 ---------------------------------------------------------------

    def projection_bounds_mm(self, proj_w: int, proj_h: int,
                             inscribed: bool = True) -> tuple:
        """投影できる範囲を mm で返す: (四隅の quad, 矩形 (x0,y0,x1,y1))。

        プロジェクターの画面4隅を机の mm 座標へ引き戻す。斜め投影なので quad は台形。

        `inscribed=True`（既定）は台形に**内接**する軸平行矩形を返す。外接矩形
        （`inscribed=False`）を使うと、台形の外側にはみ出た角の付近に UI を置いて
        しまい、実際には投影されない＝欠けて見える。実機で上端中央の文字が切れる
        のがこれ。レイアウトには必ず内接矩形を使うこと。
        """
        corners_px = np.array([[0, 0], [proj_w, 0], [proj_w, proj_h], [0, proj_h]],
                              dtype=np.float32)
        quad = self.proj_to_table(corners_px)   # TL, TR, BR, BL

        if not inscribed:
            return quad, (float(quad[:, 0].min()), float(quad[:, 1].min()),
                          float(quad[:, 0].max()), float(quad[:, 1].max()))

        # 台形の各辺の内側を採る。プロジェクター投影の台形は回転が小さいので、
        # これで実用上そのまま内接矩形になる。
        x0 = float(max(quad[0][0], quad[3][0]))   # TL / BL
        x1 = float(min(quad[1][0], quad[2][0]))   # TR / BR
        y0 = float(max(quad[0][1], quad[1][1]))   # TL / TR
        y1 = float(min(quad[2][1], quad[3][1]))   # BR / BL
        return quad, (x0, y0, x1, y1)

    def px_per_mm_at(self, x_mm: float, y_mm: float) -> float:
        """(x,y)mm 近傍での局所倍率。デバッグ表示専用（描画には使わない）。"""
        p = self.table_to_proj([[x_mm, y_mm], [x_mm + 1.0, y_mm], [x_mm, y_mm + 1.0]])
        dx = float(np.linalg.norm(p[1] - p[0]))
        dy = float(np.linalg.norm(p[2] - p[0]))
        return 0.5 * (dx + dy)


def solve_table_to_proj(points_mm: np.ndarray, points_proj: np.ndarray) -> tuple[np.ndarray, float]:
    """mm ↔ proj の対応点からホモグラフィを解き、(H, 残差RMS[mm]) を返す。

    残差は「proj 空間の誤差」ではなく mm へ引き戻して評価する。実寸精度の指標として
    そのまま README / 提出ドキュメントに載せられる値にしたいため。
    """
    src = np.asarray(points_mm, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(points_proj, dtype=np.float32).reshape(-1, 1, 2)
    if len(src) < 4:
        raise ValueError(f"ホモグラフィには4点以上必要です（現在 {len(src)} 点）")

    H, _mask = cv2.findHomography(src, dst, method=0)  # 全点を使う最小二乗
    if H is None:
        raise ValueError("ホモグラフィを求められませんでした。対応点の配置を確認してください。")

    back = transform_points(np.linalg.inv(H), dst.reshape(-1, 2))
    err = np.linalg.norm(back - src.reshape(-1, 2), axis=1)
    return H.astype(np.float64), float(np.sqrt(np.mean(err ** 2)))
