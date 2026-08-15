"""
avf_camera.py
AVFoundation で uid/名前から実カメラを直接キャプチャする（OpenCV VideoCapture を介さない）。

なぜ必要か:
  macOS の OpenCV VideoCapture は「index」でしかカメラを開けない。iPhone の
  Continuity Camera が現れると AVFoundation の列挙順と OpenCV の index 順がズレ、
  名前解決した index が別デバイス(iPhone)を開いてしまう事故が起きる。
  そこで AVFoundation の `deviceWithUniqueID_`（uid）で**実デバイスを直接掴む**ことで、
  iPhone がつながっていても目的の USB カメラを確実に選べるようにする。

外部 API は camera.Camera と揃える: start() / get_frame()->BGR(H,W,3) uint8 / stop() /
frame_size / with 文対応。例外は camera.CameraNotAvailableError を流用。

依存: pyobjc の AVFoundation / Quartz(CoreVideo) / CoreMedia / libdispatch / Foundation。
      いずれかが無い環境では start() 時に丁寧な例外を投げる（import 自体は通る）。
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from camera import CameraNotAvailableError  # noqa: E402  外部と例外型を統一


# ----------------------------------------------------------------------
# ピクセルバッファ → BGR numpy（行パディング bytesPerRow を必ず考慮）
# ----------------------------------------------------------------------
def pixelbuffer_to_bgr(pixel_buffer) -> np.ndarray:
    """CVPixelBuffer(32BGRA) を BGR(H,W,3) uint8 の ndarray に変換して返す（コピー済み）。

    32BGRA はメモリ上 B,G,R,A の並びなので、alpha を落とした [:, :, :3] がそのまま BGR。
    bytesPerRow は width*4 より大きい（行パディングがある）ことがあるため、
    reshape(h, bytesPerRow//4, 4)[:, :w, :3] で有効幅だけを切り出す。
    """
    import Quartz

    read_only = getattr(Quartz, "kCVPixelBufferLock_ReadOnly", 1)
    Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, read_only)
    try:
        w = int(Quartz.CVPixelBufferGetWidth(pixel_buffer))
        h = int(Quartz.CVPixelBufferGetHeight(pixel_buffer))
        bpr = int(Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer))
        base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
        if base is None or w == 0 or h == 0:
            raise CameraNotAvailableError("空のピクセルバッファを受け取りました。")
        buf = base.as_buffer(bpr * h)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr // 4, 4)
        bgr = arr[:, :w, :3].copy()  # パディングとalphaを除去し、ロック外でも安全なようcopy
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, read_only)
    return bgr


# ----------------------------------------------------------------------
# uid / 名前でのデバイス選択（index を一切使わない）
# ----------------------------------------------------------------------
def resolve_device_uid(uid: Optional[str] = None, name: Optional[str] = None,
                       enumerate_fn=None) -> Tuple[str, str]:
    """uid / name から対象カメラの (uid, localizedName) を返す。index は使わない。

    - uid 指定あり: 列挙結果に uid が存在するか確認して返す。
    - uid 無し・name あり: localizedName の部分一致（大小無視）で最初の一致を返す。
    - どちらも無し: デバイスが1つだけならそれを、複数なら候補を添えて例外。
    見つからなければ CameraNotAvailableError（候補一覧つき）。

    enumerate_fn はテスト用の注入口。既定は list_cameras.enumerate_avfoundation_devices。
    """
    if enumerate_fn is None:
        from list_cameras import enumerate_avfoundation_devices as enumerate_fn

    devices = enumerate_fn()
    if devices is None:
        raise CameraNotAvailableError(
            "AVFoundation でデバイスを列挙できません（pyobjc 未導入の可能性）。\n"
            "  uv pip install pyobjc-framework-AVFoundation"
        )
    if not devices:
        raise CameraNotAvailableError(
            "ビデオデバイスが1つも見つかりません。接続とカメラ権限を確認してください。"
        )

    def _candidates() -> str:
        return "\n".join(f"    name={nm!r}  uid={ud}" for _i, nm, ud in devices)

    if uid:
        for _idx, nm, ud in devices:
            if ud == uid:
                return ud, nm
        raise CameraNotAvailableError(
            f"指定 uid={uid!r} のデバイスが見つかりません。候補:\n{_candidates()}"
        )

    if name:
        low = name.lower()
        for _idx, nm, ud in devices:
            if low in nm.lower():
                return ud, nm
        raise CameraNotAvailableError(
            f"名前 {name!r} に一致するデバイスが見つかりません。候補:\n{_candidates()}"
        )

    # uid も name も未指定
    if len(devices) == 1:
        _idx, nm, ud = devices[0]
        return ud, nm
    raise CameraNotAvailableError(
        "camera.uid も camera.name も指定されていません。config.json の camera.name か "
        "camera.uid に下記のいずれかを設定してください:\n" + _candidates()
    )


# ----------------------------------------------------------------------
# キャプチャデリゲート（最新フレームを保持）
# ----------------------------------------------------------------------
_GRABBER_CLASS = None


def _get_grabber_class():
    """AVCaptureVideoDataOutput のデリゲートクラスを遅延生成してキャッシュする。
    （Foundation/objc の import を start() 実行時まで遅らせ、非mac環境の import 事故を避ける）"""
    global _GRABBER_CLASS
    if _GRABBER_CLASS is not None:
        return _GRABBER_CLASS

    import objc
    from Foundation import NSObject
    import CoreMedia

    class _FrameGrabber(NSObject):
        def initWithOwner_(self, owner):
            self = objc.super(_FrameGrabber, self).init()
            if self is None:
                return None
            self._owner = owner
            return self

        # captureOutput:didOutputSampleBuffer:fromConnection:
        def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample_buffer, connection):
            try:
                image_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
                if image_buffer is None:
                    return
                frame = pixelbuffer_to_bgr(image_buffer)
                self._owner._set_frame(frame)
            except Exception as exc:  # デリゲート内例外はキャプチャを殺さないよう保持のみ
                self._owner._set_error(exc)

    _GRABBER_CLASS = _FrameGrabber
    return _GRABBER_CLASS


# ----------------------------------------------------------------------
# 解像度プリセット選択
# ----------------------------------------------------------------------
def _preset_for(width: int, height: int):
    import AVFoundation as AVF
    table = {
        (1920, 1080): "AVCaptureSessionPreset1920x1080",
        (1280, 720): "AVCaptureSessionPreset1280x720",
        (640, 480): "AVCaptureSessionPreset640x480",
        (960, 540): "AVCaptureSessionPreset960x540",
        (3840, 2160): "AVCaptureSessionPreset3840x2160",
    }
    name = table.get((int(width), int(height)), "AVCaptureSessionPresetHigh")
    return getattr(AVF, name, getattr(AVF, "AVCaptureSessionPresetHigh"))


# ----------------------------------------------------------------------
# AVFCamera 本体
# ----------------------------------------------------------------------
class AVFCamera:
    def __init__(self, uid: Optional[str] = None, name: Optional[str] = None,
                 width: int = 1280, height: int = 720, controls: Optional[dict] = None):
        self.uid = uid
        self.name = name
        self.width = width
        self.height = height
        self.controls = controls or {}

        self._session = None
        self._grabber = None
        self._queue = None
        self._device = None

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._error: Optional[Exception] = None
        self._started = False

    # -- デリゲートからの受け口（別スレッド） --
    def _set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest = frame

    def _set_error(self, exc: Exception) -> None:
        with self._lock:
            self._error = exc

    # -- ライフサイクル --
    def start(self) -> None:
        if self._started:
            return
        try:
            import AVFoundation as AVF
            import Quartz
            import libdispatch
        except Exception as exc:
            raise CameraNotAvailableError(
                "AVFoundation バックエンドに必要な pyobjc フレームワークをインポートできません。\n"
                "  uv pip install pyobjc-framework-AVFoundation pyobjc-framework-libdispatch\n"
                f"  元エラー: {exc!r}"
            )

        uid, resolved_name = resolve_device_uid(self.uid, self.name)
        device = AVF.AVCaptureDevice.deviceWithUniqueID_(uid)
        if device is None:
            raise CameraNotAvailableError(
                f"uid={uid!r}({resolved_name!r}) のデバイスを開けませんでした。"
                " カメラ権限（システム設定 > プライバシーとセキュリティ > カメラ）を確認してください。"
            )
        self._device = device
        print(f"[avf_camera] 選択デバイス: name={resolved_name!r} uid={uid}")

        session = AVF.AVCaptureSession.alloc().init()
        preset = _preset_for(self.width, self.height)
        if session.canSetSessionPreset_(preset):
            session.setSessionPreset_(preset)

        input_obj, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if input_obj is None:
            raise CameraNotAvailableError(
                f"AVCaptureDeviceInput の生成に失敗: {err}. カメラ権限/占有を確認してください。"
            )
        if not session.canAddInput_(input_obj):
            raise CameraNotAvailableError("セッションに入力デバイスを追加できませんでした。")
        session.addInput_(input_obj)

        output = AVF.AVCaptureVideoDataOutput.alloc().init()
        output.setAlwaysDiscardsLateVideoFrames_(True)
        output.setVideoSettings_({
            Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA
        })

        grabber_cls = _get_grabber_class()
        self._grabber = grabber_cls.alloc().initWithOwner_(self)
        self._queue = libdispatch.dispatch_queue_create(b"avfcamera.frames", None)
        output.setSampleBufferDelegate_queue_(self._grabber, self._queue)

        if not session.canAddOutput_(output):
            raise CameraNotAvailableError("セッションに出力を追加できませんでした。")
        session.addOutput_(output)

        self._apply_controls(device)

        session.startRunning()
        self._session = session
        self._started = True

        # ウォームアップ: 最初の有効フレームを最大 ~4 秒待つ
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return
                if self._error is not None:
                    err = self._error
                    self._error = None
                    self.stop()
                    raise CameraNotAvailableError(f"フレーム取得中にエラー: {err!r}")
            time.sleep(0.02)
        self.stop()
        raise CameraNotAvailableError(
            "ウォームアップ中にフレームを取得できませんでした（~4秒）。"
            " カメラ権限・接続・他アプリによる占有を確認してください。"
        )

    def _apply_controls(self, device) -> None:
        """露出/フォーカス/fps を best-effort で適用する。非対応は黙って無視（適用有無は print）。"""
        import AVFoundation as AVF
        import CoreMedia

        c = self.controls
        if not c:
            return

        locked = False
        try:
            res = device.lockForConfiguration_(None)
            ok = res[0] if isinstance(res, (tuple, list)) else bool(res)
            if not ok:
                print("[avf_camera] lockForConfiguration に失敗。controls をスキップします。")
                return
            locked = True

            # --- 露出（モーションブラー対策: exposure=シャッター秒 を短く, gain=ISO） ---
            if c.get("auto_exposure") is True:
                continuous = getattr(AVF, "AVCaptureExposureModeContinuousAutoExposure", None)
                auto = getattr(AVF, "AVCaptureExposureModeAutoExpose", None)
                mode = None
                if continuous is not None and device.isExposureModeSupported_(continuous):
                    mode = continuous
                elif auto is not None and device.isExposureModeSupported_(auto):
                    mode = auto
                if mode is not None:
                    device.setExposureMode_(mode)
                    print("[avf_camera] 連続自動露出を有効にしました。")
                else:
                    print("[avf_camera] 自動露出モードはこのカメラで利用できません。")
            elif c.get("auto_exposure") is False:
                exposure = c.get("exposure")   # シャッター時間[秒]。短いほどブレに強い（例 0.002=1/500）
                gain = c.get("gain")           # ISO。短シャッターで暗くなる分を上げる
                custom = getattr(AVF, "AVCaptureExposureModeCustom", None)
                if (custom is not None and device.isExposureModeSupported_(custom)
                        and hasattr(device, "setExposureModeCustomWithDuration_ISO_completionHandler_")
                        and (exposure is not None or gain is not None)):
                    fmt = device.activeFormat()
                    # シャッター時間: 未指定なら現状維持、指定ならデバイス対応範囲にクランプ
                    dmin = dmax = None
                    if exposure is not None:
                        try:
                            dmin = CoreMedia.CMTimeGetSeconds(fmt.minExposureDuration())
                            dmax = CoreMedia.CMTimeGetSeconds(fmt.maxExposureDuration())
                            exp_c = max(dmin, min(float(exposure), dmax))
                        except Exception:
                            exp_c = float(exposure)
                        duration = CoreMedia.CMTimeMakeWithSeconds(exp_c, 1_000_000)
                    else:
                        duration = getattr(AVF, "AVCaptureExposureDurationCurrent")
                    # ISO: 未指定なら現状維持、指定なら範囲にクランプ
                    imin = imax = None
                    if gain is not None:
                        try:
                            imin = float(fmt.minISO()); imax = float(fmt.maxISO())
                            iso_c = max(imin, min(float(gain), imax))
                        except Exception:
                            iso_c = float(gain)
                    else:
                        iso_c = getattr(AVF, "AVCaptureISOCurrent")
                    try:
                        device.setExposureModeCustomWithDuration_ISO_completionHandler_(duration, iso_c, None)
                        rng = ""
                        if dmin is not None:
                            rng += f" (シャッター対応 {dmin:.5g}..{dmax:.5g}s)"
                        if imin is not None:
                            rng += f" (ISO対応 {imin:.0f}..{imax:.0f})"
                        print(f"[avf_camera] 露出カスタム適用: shutter={exposure}s ISO={gain}{rng}")
                    except Exception as e:
                        print(f"[avf_camera] 露出カスタム非対応/失敗: {e!r}")
                else:
                    locked_mode = getattr(AVF, "AVCaptureExposureModeLocked", None)
                    if locked_mode is not None and device.isExposureModeSupported_(locked_mode):
                        device.setExposureMode_(locked_mode)
                        print("[avf_camera] 露出をロックしました。")

            # --- フォーカス ---
            if c.get("autofocus") is False:
                focus = c.get("focus")
                if (focus is not None
                        and hasattr(device, "setFocusModeLockedWithLensPosition_completionHandler_")
                        and device.isFocusModeSupported_(getattr(AVF, "AVCaptureFocusModeLocked", 0))):
                    try:
                        device.setFocusModeLockedWithLensPosition_completionHandler_(float(focus), None)
                        print(f"[avf_camera] フォーカスをロック: lensPosition={focus}")
                    except Exception as e:
                        print(f"[avf_camera] フォーカスロック失敗: {e!r}")
                else:
                    locked_focus = getattr(AVF, "AVCaptureFocusModeLocked", None)
                    if locked_focus is not None and device.isFocusModeSupported_(locked_focus):
                        device.setFocusMode_(locked_focus)
                        print("[avf_camera] フォーカスをロックしました。")

            # --- fps ---
            fps = c.get("fps")
            if fps:
                try:
                    dur = CoreMedia.CMTimeMake(1, int(fps))
                    device.setActiveVideoMinFrameDuration_(dur)
                    device.setActiveVideoMaxFrameDuration_(dur)
                    print(f"[avf_camera] fps={fps} を適用しました。")
                except Exception as e:
                    print(f"[avf_camera] fps 設定失敗: {e!r}")
        except Exception as exc:
            print(f"[avf_camera] controls 適用中に例外（無視して続行）: {exc!r}")
        finally:
            if locked:
                try:
                    device.unlockForConfiguration()
                except Exception:
                    pass

    def get_frame(self) -> np.ndarray:
        if not self._started:
            raise CameraNotAvailableError("start() を呼んでからフレームを取得してください。")
        # 直近フレームが来るまで少し待つ（デリゲートは別スレッド）
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._error is not None:
                    err = self._error
                    self._error = None
                    raise CameraNotAvailableError(f"フレーム取得エラー: {err!r}")
                if self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.005)
        raise CameraNotAvailableError("フレームを取得できませんでした（タイムアウト）。")

    @property
    def frame_size(self) -> Optional[tuple]:
        with self._lock:
            if self._latest is not None:
                h, w = self._latest.shape[:2]
                return (w, h)
        return None

    def stop(self) -> None:
        if self._session is not None:
            try:
                self._session.stopRunning()
            except Exception:
                pass
        self._session = None
        self._grabber = None
        self._queue = None
        self._device = None
        self._started = False

    def __enter__(self) -> "AVFCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
