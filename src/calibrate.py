"""
calibrate.py
手動4隅キャリブレーション。

プロジェクター側に大きな基準矩形（4隅にマーカー）を全画面表示し、ユーザーは
RGB ウェブカメラの映像上で「投影された4隅」を順番にクリックする。
カメラ画像上の4点とプロジェクター座標上の4点から `cv2.findHomography` で
H_cam→proj を計算し、calibration.json に保存する。

操作:
  - 左上(TL) → 右上(TR) → 右下(BR) → 左下(BL) の順にカメラ映像ウィンドウをクリック
  - 'r' : クリックをやり直す
  - ESC / 'q' : 保存せず終了
  - 4点クリック後、's' で保存して終了（誤クリック防止のため確認ステップを挟む）

実行: uv run python src/calibrate.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from camera import Camera, CameraNotAvailableError  # noqa: E402
from projector_window import ProjectorWindow  # noqa: E402

CORNER_LABELS = ["1: 左上 (TL)", "2: 右上 (TR)", "3: 右下 (BR)", "4: 左下 (BL)"]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_target_frame(width: int, height: int, margin_ratio: float) -> tuple:
    """プロジェクターに表示する基準矩形パターンと、その4隅座標(TL,TR,BR,BL)を返す。"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    mx = int(width * margin_ratio)
    my = int(height * margin_ratio)
    tl = (mx, my)
    tr = (width - mx, my)
    br = (width - mx, height - my)
    bl = (mx, height - my)
    corners = [tl, tr, br, bl]

    cv2.rectangle(frame, tl, br, (255, 255, 255), 3)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, (pt, color) in enumerate(zip(corners, colors)):
        cv2.circle(frame, pt, 14, color, -1)
        cv2.putText(frame, str(i + 1), (pt[0] - 8, pt[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Calibration Target", (mx, my - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    proj_corners = np.array(corners, dtype=np.float32)
    return frame, proj_corners


class ClickCollector:
    def __init__(self):
        self.points = []

    def reset(self):
        self.points = []

    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((float(x), float(y)))


def draw_overlay(color_bgr: np.ndarray, collector: ClickCollector) -> np.ndarray:
    overlay = color_bgr.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for i, pt in enumerate(collector.points):
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(overlay, p, 8, colors[i % 4], -1)
        cv2.putText(overlay, str(i + 1), (p[0] + 10, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i % 4], 2, cv2.LINE_AA)

    if len(collector.points) < 4:
        msg = f"クリックしてください: {CORNER_LABELS[len(collector.points)]}"
        color = (0, 255, 255)
    else:
        msg = "4点取得済み。's'で保存 / 'r'でやり直し / ESCで中止"
        color = (0, 255, 0)
    cv2.putText(overlay, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(overlay, "投影された基準矩形の頂点をこの順にクリック: 1(赤)->2(緑)->3(青)->4(黄)",
                (20, overlay.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="手動4隅キャリブレーション")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config.json"))
    parser.add_argument("--display", type=int, default=None,
                        help="config の projector.display_index を上書きする")
    parser.add_argument("--auto-exposure", action="store_true",
                        help="カメラを自動露出にして開く（映像が暗すぎるとき）")
    parser.add_argument("--mode", default=None, choices=["borderless", "fullscreen"],
                        help="投影ウィンドウの方式。既定は borderless（macOS ではこちらが確実）")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    project_root = os.path.dirname(config_path)

    cam_cfg = cfg.get("camera", {})
    proj_cfg = cfg.get("projector", {})
    calib_path = os.path.join(project_root, cfg.get("calibration_file", "calibration.json"))

    target_frame, proj_corners = build_target_frame(
        proj_cfg.get("width", 1920), proj_cfg.get("height", 1080),
        proj_cfg.get("margin_ratio", 0.1),
    )

    if args.auto_exposure:
        cam_cfg = dict(cam_cfg, auto_exposure=True)

    win = ProjectorWindow(
        display_index=args.display if args.display is not None else proj_cfg.get("display_index", 1),
        width=proj_cfg.get("width", 1920),
        height=proj_cfg.get("height", 1080),
        fullscreen=proj_cfg.get("fullscreen", True),
        mode=args.mode or proj_cfg.get("mode", "borderless"),
        above_menu_bar=proj_cfg.get("above_menu_bar"),
    )

    try:
        cam = Camera(
            index=int(cam_cfg.get("index", 0)),
            width=int(cam_cfg.get("width", 1280)),
            height=int(cam_cfg.get("height", 720)),
            controls=cam_cfg,
        )
        cam.start()
    except CameraNotAvailableError as e:
        print(f"[calibrate] カメラを起動できません: {e}")
        return 1

    # OpenCV のウィンドウを先に作ってから pygame を開く。逆順にすると、macOS では
    # 後から作られた OpenCV(Cocoa) のウィンドウが前面を取り、プロジェクター側の
    # 全画面ウィンドウが背面に回って「投影されない」ように見えることがある。
    window_name = "calibrate - click camera image"
    cv2.namedWindow(window_name)
    collector = ClickCollector()
    cv2.setMouseCallback(window_name, collector.on_mouse, None)
    cv2.waitKey(1)

    win.open()

    print("=== 手動4隅キャリブレーション ===")
    print("プロジェクター画面に表示された基準矩形の4隅を、カメラ映像ウィンドウ上で")
    print("1(赤/左上) -> 2(緑/右上) -> 3(青/右下) -> 4(黄/左下) の順にクリックしてください。")
    print("'r': やり直し / 's': 保存して終了 / ESCか'q': 中止")

    saved = False
    try:
        while True:
            if not win.show(target_frame):
                print("[calibrate] プロジェクター側で終了要求。中止します。")
                break

            color_bgr = cam.get_frame()
            overlay = draw_overlay(color_bgr, collector)
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("[calibrate] 中止しました（保存していません）。")
                break
            if key == ord("r"):
                collector.reset()
                print("[calibrate] クリックをリセットしました。")
            if key == ord("s") and len(collector.points) == 4:
                cam_pts = np.array(collector.points, dtype=np.float32)
                H, mask = cv2.findHomography(cam_pts, proj_corners, method=0)
                if H is None:
                    print("[calibrate] ホモグラフィ計算に失敗しました。4点の取り方を見直してください。")
                    continue
                calib = {
                    "homography_cam_to_proj": H.tolist(),
                    "cam_points": cam_pts.tolist(),
                    "proj_points": proj_corners.tolist(),
                    "camera": {"width": cam_cfg.get("width", 1280), "height": cam_cfg.get("height", 720)},
                    "projector": {"width": proj_cfg.get("width", 1920), "height": proj_cfg.get("height", 1080)},
                }
                with open(calib_path, "w", encoding="utf-8") as f:
                    json.dump(calib, f, ensure_ascii=False, indent=2)
                print(f"[calibrate] 保存しました: {calib_path}")
                saved = True
                break
    finally:
        cam.stop()
        win.close()
        cv2.destroyAllWindows()

    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
