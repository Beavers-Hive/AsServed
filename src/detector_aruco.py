"""
detector_aruco.py
RGB カメラ画像から ArUco マーカーを検出し、その4隅（カメラ画像座標）を返す。

方針転換により、深度ではなく箱の上面に貼った ArUco マーカーで位置・向きを追う。
OpenCV 4.11 の新 API（getPredefinedDictionary / DetectorParameters / ArucoDetector）を使う。

`detect(bgr) -> Quad(corners[4], center, angle) | None`
  - corners は cv2.aruco が返す順序 (TL, TR, BR, BL：マーカー自身の向き基準) のまま。
    これは content.py の正方形キャンバス頂点順 (TL, TR, BR, BL) と一致するので、
    そのまま homography を組めば箱（マーカー）の回転に映像が正しく追従する。
  - 設定 id のマーカーを優先。無ければ検出された中で最大面積のものを採用。
  - EMA で軽く平滑化（ArUco はもともと安定しているので alpha は高めでよい）。
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclasses.dataclass
class Quad:
    """検出されたマーカーの矩形。corners は (4,2) float32, 順序 TL,TR,BR,BL。"""
    corners: np.ndarray
    center: tuple
    angle: float  # マーカー上辺(TL->TR)の画像上の角度[度]。0=右向き水平。


def _resolve_dictionary(name: str):
    const = getattr(cv2.aruco, name, None)
    if const is None:
        raise ValueError(
            f"未知の ArUco 辞書名: {name!r}. 例: DICT_4X4_50, DICT_5X5_100 など。"
        )
    return cv2.aruco.getPredefinedDictionary(const)


def _polygon_area(pts: np.ndarray) -> float:
    """(4,2) の多角形の面積（shoelace）。"""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


class ArucoDetector:
    def __init__(self, config: dict):
        aruco_cfg = config.get("aruco", {})
        self.dict_name = aruco_cfg.get("dict", "DICT_4X4_50")
        self.target_id = int(aruco_cfg.get("id", 0))
        self.ema_alpha = float(aruco_cfg.get("ema_alpha", 0.6))

        self._dictionary = _resolve_dictionary(self.dict_name)
        self._params = cv2.aruco.DetectorParameters()
        self._apply_tuned_params(aruco_cfg)
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)

        # 照明ムラに強くするための CLAHE（コントラスト均一化）。既定 OFF。
        self.use_clahe = bool(aruco_cfg.get("clahe", False))
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if self.use_clahe else None

        self._prev_corners: Optional[np.ndarray] = None
        self._prev_by_id: dict[int, np.ndarray] = {}

    def _apply_tuned_params(self, aruco_cfg: dict) -> None:
        """検知率を上げる既定値を入れ、config の aruco.detector_params で個別上書きを許す。"""
        p = self._params
        # サブピクセルのコーナー精緻化（ブレを減らし追従を安定させる、最も効果大）
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 30
        p.cornerRefinementMinAccuracy = 0.05
        # 適応二値化の窓サイズ範囲を広げる（明るさ/距離の変動に強く）
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 35
        p.adaptiveThreshWinSizeStep = 4
        # 小さめ/遠めのマーカーも拾う
        p.minMarkerPerimeterRate = 0.02
        p.maxMarkerPerimeterRate = 4.0
        # 斜め(透視ゆがみ)の輪郭も受け入れる
        p.polygonalApproxAccuracyRate = 0.05
        # ビット誤りの許容を少し増やす（IDでフィルタするため誤検出リスクは低い）
        p.errorCorrectionRate = 0.7

        # config で任意の DetectorParameters を上書き可能に（例: {"minMarkerPerimeterRate": 0.03}）
        for k, v in (aruco_cfg.get("detector_params") or {}).items():
            if hasattr(p, k):
                setattr(p, k, v)
            else:
                print(f"[detector_aruco] 未知の DetectorParameters キーを無視: {k!r}")

    def reset(self) -> None:
        self._prev_corners = None
        self._prev_by_id = {}

    def detect(self, bgr: np.ndarray) -> Optional[Quad]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        corners_list, ids, _rejected = self._detector.detectMarkers(gray)

        if ids is None or len(corners_list) == 0:
            self._prev_corners = None
            return None

        ids = ids.flatten()
        chosen = None

        # 1) 設定 id を優先
        for c, i in zip(corners_list, ids):
            if int(i) == self.target_id:
                chosen = c.reshape(4, 2).astype(np.float32)
                break

        # 2) 無ければ最大面積のマーカー
        if chosen is None:
            best_area = -1.0
            for c in corners_list:
                pts = c.reshape(4, 2).astype(np.float32)
                a = _polygon_area(pts)
                if a > best_area:
                    best_area = a
                    chosen = pts

        if chosen is None:
            self._prev_corners = None
            return None

        # EMA 平滑化（前フレームと同一マーカーである前提の軽い平滑化）
        if self._prev_corners is not None and self._prev_corners.shape == chosen.shape:
            a = self.ema_alpha
            smoothed = a * chosen + (1.0 - a) * self._prev_corners
        else:
            smoothed = chosen
        self._prev_corners = smoothed

        center = (float(smoothed[:, 0].mean()), float(smoothed[:, 1].mean()))
        top_edge = smoothed[1] - smoothed[0]  # TL -> TR
        angle = float(np.degrees(np.arctan2(top_edge[1], top_edge[0])))
        return Quad(corners=smoothed.astype(np.float32), center=center, angle=angle)

    def detect_all(self, bgr: np.ndarray, target_ids: Optional[Iterable[int]] = None) -> dict[int, Quad]:
        """検出した ArUco マーカーを id -> Quad で返す。

        `detect()` は既存の本番ループ用に単一 id を返す。こちらは pasta_projection.py のように
        複数 id を同時に使いたい用途向け。重複 id が見つかった場合は面積が最大のものを採用する。
        """
        target_set = None if target_ids is None else {int(i) for i in target_ids}

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        corners_list, ids, _rejected = self._detector.detectMarkers(gray)

        if ids is None or len(corners_list) == 0:
            self._prev_by_id = {}
            return {}

        best_by_id: dict[int, np.ndarray] = {}
        area_by_id: dict[int, float] = {}
        for c, i in zip(corners_list, ids.flatten()):
            marker_id = int(i)
            if target_set is not None and marker_id not in target_set:
                continue
            pts = c.reshape(4, 2).astype(np.float32)
            area = _polygon_area(pts)
            if area > area_by_id.get(marker_id, -1.0):
                best_by_id[marker_id] = pts
                area_by_id[marker_id] = area

        out: dict[int, Quad] = {}
        next_prev: dict[int, np.ndarray] = {}
        for marker_id in sorted(best_by_id):
            chosen = best_by_id[marker_id]
            prev = self._prev_by_id.get(marker_id)
            if prev is not None and prev.shape == chosen.shape:
                a = self.ema_alpha
                smoothed = a * chosen + (1.0 - a) * prev
            else:
                smoothed = chosen
            smoothed = smoothed.astype(np.float32)
            next_prev[marker_id] = smoothed

            center = (float(smoothed[:, 0].mean()), float(smoothed[:, 1].mean()))
            top_edge = smoothed[1] - smoothed[0]
            angle = float(np.degrees(np.arctan2(top_edge[1], top_edge[0])))
            out[marker_id] = Quad(corners=smoothed, center=center, angle=angle)

        self._prev_by_id = next_prev
        return out


def expand_quad(corners: np.ndarray, mult: float) -> np.ndarray:
    """4隅を各点について中心から mult 倍に相似拡大した4隅を返す（向き・順序は保持）。

    main.py と test_aruco.py で共用する。mult=1.0 で元と同じ、2.5 で外側に2.5倍。
    """
    pts = np.asarray(corners, dtype=np.float32)
    center = pts.mean(axis=0, keepdims=True)
    return (center + (pts - center) * float(mult)).astype(np.float32)
