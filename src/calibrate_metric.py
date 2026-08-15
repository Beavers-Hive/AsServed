"""
calibrate_metric.py
寸法較正: テーブル平面の mm 座標 → プロジェクター px のホモグラフィを求める。

前提として、先に既存の4隅クリック較正 (`src/calibrate.py`) で
`calibration.json`（H_cam→proj）を作っておくこと。本スクリプトは
較正ボード（make_board.py で生成・印刷したもの）を机に置いて、

    ボードのマーカー4隅(mm, 公称値)  ->  カメラで検出した4隅(px)  ->  H_cam→proj  ->  proj(px)

の対応から H_table_mm→proj を最小二乗で解く。マーカー6枚なら24点使えるので、
1枚のマーカーから外挿する場合よりも遠くまで寸法が保つ。

実行:
    # 1) 較正（ボードを机に置いて）
    uv run python src/calibrate_metric.py

    # 2) 印刷が縮んでいた場合（マーカー1辺を実測して渡す）
    uv run python src/calibrate_metric.py --measured-marker-mm 58.6

    # 3) 検証（100mm グリッドと 100mm スケールバーを投影。定規で測って一致を確認）
    uv run python src/calibrate_metric.py --verify

出力:
    metric.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from camera import Camera, CameraNotAvailableError  # noqa: E402
from detector_aruco import ArucoDetector  # noqa: E402
from geometry import TableFrame, rect_mm, solve_table_to_proj, transform_points  # noqa: E402
from projector_window import ProjectorWindow  # noqa: E402


def load_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_board_corners(cam: Camera, detector: ArucoDetector, target_ids: set[int],
                          frames: int, min_markers: int) -> dict[int, np.ndarray]:
    """複数フレームでボードを検出し、ID ごとに4隅の平均（カメラ px）を返す。

    1フレームだけだとコーナー検出のゆらぎがそのまま寸法誤差になるので平均する。
    """
    acc: dict[int, list[np.ndarray]] = {}
    for i in range(frames):
        bgr = cam.get_frame()
        found = detector.detect_all(bgr, target_ids)
        for marker_id, quad in found.items():
            acc.setdefault(marker_id, []).append(quad.corners.astype(np.float64))
        if i % 10 == 0:
            print(f"[calibrate_metric] frame {i + 1}/{frames}  検出 {len(found)} 枚", end="\r")
        time.sleep(0.01)
    print()

    # 全フレーム中の半分以上で見えていたマーカーだけ採用（ちらつきを除く）
    stable = {mid: np.mean(v, axis=0) for mid, v in acc.items() if len(v) >= frames * 0.5}
    if len(stable) < min_markers:
        raise SystemExit(
            f"[calibrate_metric] 安定して検出できたマーカーが {len(stable)} 枚しかありません"
            f"（最低 {min_markers} 枚必要）。\n"
            "  - ボード全体がカメラに写っているか\n"
            "  - 照明が足りているか / プロジェクターの光が白飛びさせていないか\n"
            "  - board_layout.json の dict と config.json の aruco.dict が一致しているか\n"
            "を確認してください。投影を消した状態で実行するのが確実です。"
        )
    return stable


def render_verify(frame: TableFrame, proj_w: int, proj_h: int, grid_mm: float,
                  extent_mm: float) -> np.ndarray:
    """定規で確かめるための検証パターン。100mm グリッド + スケールバー + 円。"""
    canvas = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)

    def line_mm(p0, p1, color, thickness=2):
        pts = frame.table_to_proj([p0, p1]).astype(np.int32)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), color, thickness, cv2.LINE_AA)

    n = int(extent_mm // grid_mm)
    for i in range(n + 1):
        v = i * grid_mm
        line_mm([v, 0], [v, n * grid_mm], (60, 60, 60), 1)
        line_mm([0, v], [n * grid_mm, v], (60, 60, 60), 1)

    # 原点まわりの軸（x=赤, y=緑）
    line_mm([0, 0], [grid_mm, 0], (0, 0, 255), 3)
    line_mm([0, 0], [0, grid_mm], (0, 255, 0), 3)

    # 100mm スケールバー（端に目盛り）
    bar_y = grid_mm * 0.5
    line_mm([grid_mm, bar_y], [grid_mm + 100.0, bar_y], (255, 255, 255), 3)
    for x in (grid_mm, grid_mm + 100.0):
        line_mm([x, bar_y - 5.0], [x, bar_y + 5.0], (255, 255, 255), 3)

    # 直径 220mm の皿の輪郭（実際のパスタ皿と重ねて確認する用）
    center = (grid_mm * 1.5 + 100.0, grid_mm * 1.5)
    ring = frame.table_to_proj(
        np.stack([
            center[0] + 110.0 * np.cos(np.linspace(0, 2 * np.pi, 128)),
            center[1] + 110.0 * np.sin(np.linspace(0, 2 * np.pi, 128)),
        ], axis=1)
    ).astype(np.int32)
    cv2.polylines(canvas, [ring.reshape(-1, 1, 2)], True, (0, 200, 255), 2, cv2.LINE_AA)

    label = frame.table_to_proj([[grid_mm, bar_y - 12.0]])[0].astype(int)
    cv2.putText(canvas, "100 mm", (int(label[0]), int(label[1])), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    label2 = frame.table_to_proj([[center[0] - 40.0, center[1]]])[0].astype(int)
    cv2.putText(canvas, "dia 220 mm", (int(label2[0]), int(label2[1])), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent

    p = argparse.ArgumentParser(description="mm → projector px の寸法較正")
    p.add_argument("--config", default=str(root / "config.json"))
    p.add_argument("--board", default=str(root / "markers" / "board_layout.json"))
    p.add_argument("--out", default=str(root / "metric.json"))
    p.add_argument("--frames", type=int, default=60, help="平均に使うフレーム数")
    p.add_argument("--min-markers", type=int, default=4, help="必要な検出マーカー枚数")
    p.add_argument("--measured-marker-mm", type=float, default=None,
                   help="印刷後に実測したマーカー1辺(mm)。公称値とずれた分だけレイアウトを相似スケールする")
    p.add_argument("--verify", action="store_true",
                   help="較正済み metric.json を使って検証パターンを投影する（較正はしない）")
    p.add_argument("--grid-mm", type=float, default=100.0, help="検証グリッドの間隔(mm)")
    p.add_argument("--extent-mm", type=float, default=400.0, help="検証グリッドの範囲(mm)")
    args = p.parse_args()

    cfg = load_json(args.config)
    proj_cfg = cfg.get("projector", {})
    proj_w = int(proj_cfg.get("width", 1920))
    proj_h = int(proj_cfg.get("height", 1080))

    # ---- 検証モード -------------------------------------------------------
    if args.verify:
        metric_path = Path(args.out)
        if not metric_path.exists():
            print(f"[calibrate_metric] {metric_path} がありません。先に --verify なしで較正してください。")
            return 1
        frame = TableFrame.load(metric_path, root / cfg.get("calibration_file", "calibration.json"))
        canvas = render_verify(frame, proj_w, proj_h, args.grid_mm, args.extent_mm)
        win = ProjectorWindow(display_index=int(proj_cfg.get("display_index", 1)),
                              width=proj_w, height=proj_h,
                              fullscreen=bool(proj_cfg.get("fullscreen", True)),
                              mode=proj_cfg.get("mode", "borderless"),
                              above_menu_bar=proj_cfg.get("above_menu_bar"))
        win.open()
        print("[calibrate_metric] 検証パターンを投影中。定規で以下を実測してください:")
        print("  - 白いスケールバー        → 100.0 mm であること")
        print("  - グリッドの1マス          → {:.1f} mm であること".format(args.grid_mm))
        print("  - オレンジの円の直径       → 220.0 mm であること（手持ちの皿と重ねるのが早い）")
        if frame.board_rms_mm is not None:
            print(f"  - 較正時の残差RMS          = {frame.board_rms_mm:.2f} mm")
        print("[calibrate_metric] ESC または Ctrl-C で終了。")
        try:
            while win.show(canvas):
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        finally:
            win.close()
        return 0

    # ---- 較正モード -------------------------------------------------------
    calib_path = root / cfg.get("calibration_file", "calibration.json")
    if not calib_path.exists():
        print(f"[calibrate_metric] {calib_path} がありません。先に `uv run python src/calibrate.py` を実行してください。")
        return 1
    H_cam_to_proj = np.array(load_json(calib_path)["homography_cam_to_proj"], dtype=np.float64)

    board_path = Path(args.board)
    if not board_path.exists():
        print(f"[calibrate_metric] {board_path} がありません。先に `uv run python src/make_board.py` を実行してください。")
        return 1
    board = load_json(board_path)

    corners_mm_by_id = {int(k): np.array(v, dtype=np.float64) for k, v in board["corners_mm"].items()}
    nominal_marker_mm = float(board["marker_mm"])
    scale = 1.0
    if args.measured_marker_mm is not None:
        scale = float(args.measured_marker_mm) / nominal_marker_mm
        corners_mm_by_id = {k: v * scale for k, v in corners_mm_by_id.items()}
        print(f"[calibrate_metric] 印刷スケール補正: 公称 {nominal_marker_mm:g}mm → "
              f"実測 {args.measured_marker_mm:g}mm (x{scale:.4f})")

    if board.get("dict") != cfg.get("aruco", {}).get("dict"):
        print(f"[calibrate_metric] 警告: ボードの辞書 {board.get('dict')!r} と "
              f"config の aruco.dict {cfg.get('aruco', {}).get('dict')!r} が違います。")

    cam_cfg = cfg.get("camera", {})
    try:
        cam = Camera(index=int(cam_cfg.get("index", 0)),
                     width=int(cam_cfg.get("width", 1280)),
                     height=int(cam_cfg.get("height", 720)),
                     controls=cam_cfg)
        cam.start()
    except CameraNotAvailableError as e:
        print(f"[calibrate_metric] カメラを起動できません:\n{e}")
        return 1

    detector = ArucoDetector(cfg)
    try:
        print(f"[calibrate_metric] ボードを机に平らに置いてください。{args.frames} フレーム平均します...")
        detected = collect_board_corners(cam, detector, set(corners_mm_by_id),
                                         args.frames, args.min_markers)
    finally:
        cam.stop()

    pts_mm, pts_proj = [], []
    for marker_id, cam_corners in sorted(detected.items()):
        proj_corners = transform_points(H_cam_to_proj, cam_corners)
        pts_mm.append(corners_mm_by_id[marker_id])
        pts_proj.append(proj_corners)
    pts_mm = np.concatenate(pts_mm, axis=0)
    pts_proj = np.concatenate(pts_proj, axis=0)

    H, rms_mm = solve_table_to_proj(pts_mm, pts_proj)

    out = {
        "homography_table_mm_to_proj": H.tolist(),
        "residual_rms_mm": rms_mm,
        "marker_ids_used": sorted(detected),
        "point_count": int(len(pts_mm)),
        "board": {
            "layout_file": str(board_path),
            "nominal_marker_mm": nominal_marker_mm,
            "measured_marker_mm": args.measured_marker_mm,
            "print_scale": scale,
        },
        "projector": {"width": proj_w, "height": proj_h},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[calibrate_metric] 保存しました: {args.out}")
    print(f"[calibrate_metric] 使用マーカー {sorted(detected)} / 対応点 {len(pts_mm)}")
    print(f"[calibrate_metric] 残差 RMS = {rms_mm:.2f} mm")
    if rms_mm > 5.0:
        print("[calibrate_metric] 警告: 残差が大きいです。ボードが浮いている/たわんでいる、"
              "カメラのピンぼけ、calibration.json のズレを疑ってください。")
    print("[calibrate_metric] 次: uv run python src/calibrate_metric.py --verify で定規チェック")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
