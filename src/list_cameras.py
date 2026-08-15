"""
list_cameras.py
接続されているビデオ入力デバイス（USB ウェブカメラ等）を列挙するユーティリティ。

ユーザーが config.json の camera.index に設定すべき番号を特定するために使う。
macOS の AVFoundation で列挙した順序が、OpenCV の CAP_AVFOUNDATION の index 順と
ほぼ一致する前提。

実行: uv run python src/list_cameras.py
（デバイス列挙にはカメラ権限が必要な場合があるため、ユーザー自身のターミナルで実行）
"""
from __future__ import annotations

from typing import List, Optional, Tuple


def enumerate_avfoundation_devices() -> Optional[List[Tuple[int, str, str]]]:
    """AVFoundation のビデオデバイスを列挙し (index, localizedName, uniqueID) を返す。

    新しめの macOS 向けに `AVCaptureDeviceDiscoverySession` を優先し、
    使えなければ従来の `AVCaptureDevice.devicesWithMediaType_` にフォールバックする。
    pyobjc 未導入時は None を返す。
    （probe_uvc_depth.py の列挙ロジックと同等。RGB ワークフローで独立して使えるよう再掲）
    """
    try:
        import AVFoundation  # type: ignore
    except Exception as exc:
        print("[list_cameras] pyobjc の AVFoundation をインポートできませんでした。")
        print("[list_cameras]   uv pip install pyobjc-framework-AVFoundation")
        print(f"[list_cameras]   元エラー: {exc!r}")
        return None

    media_type = AVFoundation.AVMediaTypeVideo
    devices = None

    try:
        candidate_type_names = [
            "AVCaptureDeviceTypeExternal",
            "AVCaptureDeviceTypeExternalUnknown",
            "AVCaptureDeviceTypeBuiltInWideAngleCamera",
            "AVCaptureDeviceTypeContinuityCamera",
        ]
        device_types = [getattr(AVFoundation, n) for n in candidate_type_names
                        if getattr(AVFoundation, n, None) is not None]
        if device_types:
            session = AVFoundation.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
                device_types, media_type, 0,
            )
            devices = list(session.devices())
    except Exception as exc:
        print(f"[list_cameras] DiscoverySession 失敗（フォールバック）: {exc!r}")
        devices = None

    if not devices:
        try:
            devices = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(media_type))
        except Exception as exc:
            print(f"[list_cameras] devicesWithMediaType_ も失敗: {exc!r}")
            return None

    result: List[Tuple[int, str, str]] = []
    for i, dev in enumerate(devices):
        try:
            name = str(dev.localizedName())
        except Exception:
            name = "<unknown>"
        try:
            uid = str(dev.uniqueID())
        except Exception:
            uid = "<unknown>"
        result.append((i, name, uid))
    return result


def main() -> int:
    import argparse
    argparse.ArgumentParser(description="接続ビデオデバイスを列挙する").parse_args()

    print("=== 接続ビデオデバイス一覧 ===")
    devices = enumerate_avfoundation_devices()
    if devices is None:
        print("[list_cameras] pyobjc 未導入のため列挙できませんでした。")
        return 1
    if not devices:
        print("[list_cameras] ビデオデバイスが見つかりません。接続とカメラ権限を確認してください。")
        return 1
    for idx, name, uid in devices:
        print(f"  [{idx}] {name}    uid={uid}")
    print("\nconfig.json の camera.index に番号を設定できますが、抜き差しで index が変わる場合は")
    print("camera.name に上のデバイス名（部分一致可, 例 \"16MP USB Camera\"）を入れると自動で掴みます。")
    print("さらに確実にしたいときは camera.uid に上の uid を丸ごとコピーしてください（完全一致で最優先）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
