"""
fetch_model.py
MediaPipe の手ランドマークモデル (hand_landmarker.task, 約 7.5MB) を取得する。

MediaPipe 1.0 で旧 `mp.solutions.hands`（モデル同梱）が廃止され、Tasks API と
外部モデルファイルの組み合わせになりました。ライセンスと配布サイズの都合で
リポジトリには同梱せず、各自が Google の公式配布元から取得する形にしています。

    uv run python src/fetch_model.py

取得先: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
モデル: hand_landmarker (float16) / Apache License 2.0
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/1/hand_landmarker.task")
MIN_BYTES = 1_000_000    # 取得失敗時に HTML のエラーページを掴んでいないかの下限チェック


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="MediaPipe 手ランドマークモデルを取得する")
    p.add_argument("--out", default=str(root / "models" / "hand_landmarker.task"))
    p.add_argument("--url", default=URL)
    p.add_argument("--force", action="store_true", help="既にあっても取得し直す")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"[fetch_model] 既にあります: {out} ({out.stat().st_size:,} bytes)")
        print("[fetch_model] 取り直す場合は --force を付けてください。")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_model] 取得中: {args.url}")
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with urllib.request.urlopen(args.url, timeout=60) as res, open(tmp, "wb") as f:
            total = int(res.headers.get("Content-Length", 0))
            read = 0
            while True:
                chunk = res.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total:
                    print(f"\r[fetch_model] {read:,} / {total:,} bytes "
                          f"({read * 100 // total}%)", end="")
        print()
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"[fetch_model] 取得に失敗しました: {exc!r}", file=sys.stderr)
        print(f"[fetch_model] ブラウザで {args.url} を保存し、{out} に置いても構いません。",
              file=sys.stderr)
        return 1

    size = tmp.stat().st_size
    if size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        print(f"[fetch_model] 取得したファイルが小さすぎます ({size:,} bytes)。"
              "URL かネットワークを確認してください。", file=sys.stderr)
        return 1

    tmp.replace(out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"[fetch_model] 保存しました: {out} ({size:,} bytes)")
    print(f"[fetch_model] sha256 = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
