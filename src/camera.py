"""
camera.py
RGB カメラ入力のファサード。バックエンドを切り替えられる:

  - backend="avf"（既定）: AVFoundation で uid/名前から実デバイスを直接キャプチャする
    （src/avf_camera.py）。macOS の OpenCV VideoCapture は index でしか開けず、
    iPhone(Continuity Camera) が現れると列挙順と index 順がズレて別デバイスを掴む
    事故が起きるため、uid/名前指定で確実に目的のカメラを開くこの方式を既定にする。
  - backend="opencv": 従来の cv2.VideoCapture 実装（index/名前解決つき）。フォールバック用。

外部 API（start/get_frame/stop/frame_size/with 文）はどちらのバックエンドでも不変。

使い方:
    cam = Camera(controls=cfg["camera"])   # backend/uid/name/width/height を config から取る
    cam.start()
    bgr = cam.get_frame()   # BGR(H,W,3) uint8
    cam.stop()
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class CameraNotAvailableError(RuntimeError):
    """カメラを開けない / フレームを取得できない場合に送出する。"""


class Camera:
    """バックエンドを切り替えるファサード。

    backend/uid/name は明示 kwarg が最優先、無ければ controls（=camera config）から取る。
    controls も指定も無い場合は backend="avf" 既定。
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 controls: Optional[dict] = None,
                 backend: Optional[str] = None,
                 uid: Optional[str] = None, name: Optional[str] = None):
        self.index = index
        self.width = width
        self.height = height
        self.controls = controls or {}
        self.backend = (backend or self.controls.get("backend") or "avf").lower()
        self.uid = uid if uid is not None else self.controls.get("uid")
        self.name = name if name is not None else self.controls.get("name")
        self._impl = None

    def start(self) -> None:
        if self._impl is not None:
            return
        if self.backend == "avf":
            import os
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from avf_camera import AVFCamera
            self._impl = AVFCamera(
                uid=self.uid, name=self.name,
                width=self.width, height=self.height, controls=self.controls,
            )
        elif self.backend == "opencv":
            self._impl = _OpenCVCamera(
                index=self.index, width=self.width, height=self.height,
                controls=self.controls,
            )
        else:
            raise CameraNotAvailableError(
                f"未知の camera.backend={self.backend!r}. 'avf' か 'opencv' を指定してください。"
            )
        self._impl.start()

    def get_frame(self) -> np.ndarray:
        if self._impl is None:
            raise CameraNotAvailableError("start() を呼んでからフレームを取得してください。")
        return self._impl.get_frame()

    @property
    def frame_size(self) -> Optional[tuple]:
        return None if self._impl is None else self._impl.frame_size

    def stop(self) -> None:
        if self._impl is not None:
            self._impl.stop()
        self._impl = None

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class _OpenCVCamera:
    """従来の cv2.VideoCapture 実装（index ベース、名前/uid で index を解決）。フォールバック用。"""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 controls: Optional[dict] = None):
        self.index = index
        self.width = width
        self.height = height
        self.controls = controls or {}
        self._cap = None
        self._started = False

    def _resolve_index(self) -> int:
        """camera.name / camera.uid が指定されていれば AVFoundation を列挙して一致 index を返す。
        指定が無い/見つからない場合は index にフォールバック（※ここが iPhone でズレうる箇所）。"""
        want_name = self.controls.get("name")
        want_uid = self.controls.get("uid")
        if not want_name and not want_uid:
            return self.index

        import os
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from list_cameras import enumerate_avfoundation_devices
            devices = enumerate_avfoundation_devices()
        except Exception as exc:
            print(f"[camera:opencv] デバイス列挙に失敗（index={self.index} を使用）: {exc!r}")
            return self.index
        if not devices:
            print(f"[camera:opencv] デバイスを列挙できず、index={self.index} を使用します。")
            return self.index

        if want_uid:
            for i, name, uid in devices:
                if uid == want_uid:
                    print(f"[camera:opencv] uid 一致 → index={i} ({name})")
                    return i
        if want_name:
            for i, name, uid in devices:
                if want_name.lower() in name.lower():
                    print(f"[camera:opencv] name '{want_name}' 一致 → index={i} ({name})")
                    return i

        avail = ", ".join(f"[{i}]{name}" for i, name, uid in devices)
        print(f"[camera:opencv] 指定カメラ(name={want_name!r}, uid={want_uid!r})が見つかりません。"
              f"index={self.index} を使用。候補: {avail}")
        return self.index

    def _apply_controls(self, cap, cv2) -> None:
        c = self.controls
        applied = []
        if c.get("autofocus") is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if c["autofocus"] else 0)
            applied.append(("AUTOFOCUS", cap.get(cv2.CAP_PROP_AUTOFOCUS)))
        if c.get("focus") is not None:
            cap.set(cv2.CAP_PROP_FOCUS, float(c["focus"]))
            applied.append(("FOCUS", cap.get(cv2.CAP_PROP_FOCUS)))
        if c.get("auto_exposure") is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if c["auto_exposure"] else 0.25)
            applied.append(("AUTO_EXPOSURE", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)))
        # Some UVC drivers interpret EXPOSURE very differently. Writing it while
        # auto exposure is enabled can leave the camera almost black.
        manual_exposure = c.get("auto_exposure") is False
        if manual_exposure and c.get("exposure") is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(c["exposure"]))
            applied.append(("EXPOSURE", cap.get(cv2.CAP_PROP_EXPOSURE)))
        if manual_exposure and c.get("gain") is not None:
            cap.set(cv2.CAP_PROP_GAIN, float(c["gain"]))
            applied.append(("GAIN", cap.get(cv2.CAP_PROP_GAIN)))
        if applied:
            print("[camera:opencv] 手動制御を適用(読み戻し値):",
                  ", ".join(f"{k}={v:.3g}" for k, v in applied))

    def start(self) -> None:
        import time
        import cv2

        if self._started:
            return
        self.index = self._resolve_index()
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            raise CameraNotAvailableError(
                f"カメラ index={self.index} を開けませんでした。\n"
                "  - USB ウェブカメラが接続されているか\n"
                "  - macOS の『システム設定 > プライバシーとセキュリティ > カメラ』で\n"
                "    ターミナル(またはIDE)にカメラ権限が付与されているか\n"
                "  - `uv run python src/list_cameras.py` で正しい index を確認したか\n"
                "  を確認してください。"
            )
        fourcc = self.controls.get("fourcc")
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(fourcc)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        fps = self.controls.get("fps")
        if fps:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        self._apply_controls(cap, cv2)

        warmed = False
        for _ in range(80):
            ok, frame = cap.read()
            if ok and frame is not None:
                warmed = True
                break
            time.sleep(0.05)
        if not warmed:
            cap.release()
            raise CameraNotAvailableError(
                f"カメラ index={self.index} は開けましたが、フレームを取得できませんでした。\n"
                "  - `uv run python src/list_cameras.py` で USB ウェブカメラの index を確認し、\n"
                "    config.json の camera.index を合わせてください\n"
                "  - 他アプリがカメラを占有していないか確認してください"
            )
        self._cap = cap
        self._started = True

        fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fcc_str = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") or str(fcc)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[camera:opencv] index={self.index} 開始: {w}x{h} fourcc={fcc_str} "
              f"fps(report)={cap.get(cv2.CAP_PROP_FPS):.0f}")

    def get_frame(self) -> np.ndarray:
        import time

        if not self._started or self._cap is None:
            raise CameraNotAvailableError("start() を呼んでからフレームを取得してください。")
        for _ in range(10):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.02)
        raise CameraNotAvailableError(
            "フレームを取得できませんでした。ケーブル接続やカメラ権限、"
            "他アプリによる占有を確認してください。"
        )

    @property
    def frame_size(self) -> Optional[tuple]:
        if self._cap is None:
            return None
        import cv2
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._started = False

    def __enter__(self) -> "_OpenCVCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
