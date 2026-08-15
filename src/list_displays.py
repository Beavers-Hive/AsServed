"""
list_displays.py
接続されているディスプレイを列挙し、どれがプロジェクターかを目で確かめる。

「投影されない」ときの原因はだいたい次のどれかで、このツールで切り分けられる。

  1. `projector.display_index` が違う（プロジェクターが 0 番だった、など）
  2. `projector.width/height` が実際の解像度と違う
     → 投影はされているが位置がずれる／端が切れる。較正が根本的に狂う
  3. ミラーリング設定になっていて、そもそも2画面になっていない
     → 検出ディスプレイ数が 1 と表示される

実行:
    uv run python src/list_displays.py             # 一覧を表示するだけ
    uv run python src/list_displays.py --test      # 各ディスプレイに番号を順に表示
    uv run python src/list_displays.py --test --display 1 --seconds 8
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description="ディスプレイの列挙と投影先の確認")
    p.add_argument("--test", action="store_true", help="番号入りのテストパターンを実際に表示する")
    p.add_argument("--display", type=int, default=None, help="このディスプレイだけ試す")
    p.add_argument("--seconds", type=float, default=4.0, help="1画面あたりの表示秒数")
    args = p.parse_args()

    import cv2
    import pygame

    pygame.init()
    sizes = pygame.display.get_desktop_sizes()
    print(f"[list_displays] 検出したディスプレイ: {len(sizes)} 台")
    for i, (w, h) in enumerate(sizes):
        print(f"  display_index={i}  {w} x {h}")

    if len(sizes) == 1:
        print("[list_displays] 1台しか見えていません。プロジェクターがミラーリングになっているか、")
        print("[list_displays] 接続されていない可能性があります。システム設定 > ディスプレイ で")
        print("[list_displays] 『ディスプレイを拡張』になっているか確認してください。")

    print("[list_displays] config.json の projector.width/height は、使うディスプレイの")
    print("[list_displays] 上記サイズと一致させてください（違うと較正が狂います）。")

    if not args.test:
        pygame.quit()
        return 0

    indices = [args.display] if args.display is not None else list(range(len(sizes)))
    for idx in indices:
        if not (0 <= idx < len(sizes)):
            print(f"[list_displays] display_index={idx} は存在しません。")
            continue
        w, h = sizes[idx]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(frame, (4, 4), (w - 5, h - 5), (255, 255, 255), 6)
        cv2.putText(frame, str(idx), (w // 2 - 120, h // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 10.0, (255, 255, 255), 20, cv2.LINE_AA)
        cv2.putText(frame, f"display_index={idx}  {w}x{h}", (40, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3, cv2.LINE_AA)

        screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN | pygame.NOFRAME, display=idx)
        surface = pygame.surfarray.make_surface(np.transpose(frame[:, :, ::-1], (1, 0, 2)))
        print(f"[list_displays] display_index={idx} に表示中 ({args.seconds:.0f}秒)...")

        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return 0
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            time.sleep(0.03)

    pygame.quit()
    print("[list_displays] プロジェクターに数字が出たディスプレイの番号を、")
    print("[list_displays] config.json の projector.display_index に設定してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
