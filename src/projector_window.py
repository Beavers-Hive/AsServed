"""
projector_window.py
生成コンテンツをプロジェクター（2画面目などの指定ディスプレイ）に全画面表示する。

macOS では OpenCV の `cv2.imshow` + フルスクリーンは2画面目で不安定になりやすいため、
SDL2 ベースの pygame を使い `pygame.display.set_mode(..., display=idx)` で
表示先ディスプレイを明示的に指定する。

使い方:
    win = ProjectorWindow(display_index=1, width=1920, height=1080, fullscreen=True)
    win.open()
    while True:
        frame_bgr = ...  # (H, W, 3) uint8
        if not win.show(frame_bgr):
            break  # ESC/終了要求
    win.close()
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional

import numpy as np

# SDL は「フルスクリーンのウィンドウがフォーカスを失ったら最小化する」のが既定。
# このアプリでは操作用の OpenCV ウィンドウをクリックした瞬間に投影が消えてしまうので、
# 明示的に無効化する。pygame(SDL) の初期化より前に環境変数で渡す必要がある。
os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")
# macOS の「フルスクリーン＝専用 Space」を使わせない。mode="fullscreen" を選んだときに
# 他ウィンドウをクリックしても投影が引っ込まなくなる。
os.environ.setdefault("SDL_VIDEO_MAC_FULLSCREEN_SPACES", "0")

NS_STATUS_WINDOW_LEVEL = 25   # メニューバー(NSMainMenuWindowLevel=24)より上


def _raise_above_menu_bar(pygame) -> bool:
    """macOS で、この pygame ウィンドウをメニューバーより前面に出す。

    枠なしウィンドウはメニューバーの下に潜るため、投影面の上端に macOS の
    ステータスバーが焼き込まれてしまう。NSWindow のウィンドウレベルを
    メニューバーより上げると、フルスクリーンにせずに隠せる。
    失敗しても投影自体は動くので、例外は握って False を返す。
    """
    if sys.platform != "darwin":
        return False
    try:
        import objc  # pyobjc（AVFoundation 用に既に依存として入っている）

        capsule = pygame.display.get_wm_info()["window"]
        ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
        ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(
            capsule, ctypes.pythonapi.PyCapsule_GetName(capsule))

        window = objc.objc_object(c_void_p=ptr)
        window.setLevel_(NS_STATUS_WINDOW_LEVEL)
        return True
    except Exception as exc:
        print(f"[projector_window] メニューバーより前面に出せませんでした: {exc!r}")
        return False


class ProjectorWindow:
    """指定ディスプレイを覆うウィンドウ。

    mode:
      "borderless" (既定) … デスクトップと同じ大きさの枠なしウィンドウでディスプレイを覆う。
                            macOS ではこちらが確実。真のフルスクリーンにすると
                            専用 Space が作られ、他ウィンドウをクリックした瞬間に
                            投影が引っ込む（「一瞬映って消える」の正体）。
      "fullscreen"        … 従来どおり SDL の FULLSCREEN。他OSや必要時のフォールバック。
    """

    def __init__(self, display_index: int = 0, width: int = 1920, height: int = 1080,
                 fullscreen: bool = True, mode: str = "borderless",
                 above_menu_bar: Optional[bool] = None):
        self.display_index = display_index
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.mode = mode
        # None = 自動（2画面目に出すときだけ前面化する）。主画面で常に前面化すると
        # 操作用のウィンドウが隠れて何もできなくなるので、既定は自動。
        self.above_menu_bar = above_menu_bar
        self._pygame = None
        self._screen = None
        self._clock = None
        self._opened = False
        self._surface_size = (width, height)
        self._key_buffer: list = []
        self._quit_requested = False

    def open(self) -> None:
        import pygame  # 遅延importでdevice不要な用途への影響を避ける

        pygame.init()
        num_displays = pygame.display.get_num_displays() if hasattr(
            pygame.display, "get_num_displays"
        ) else len(pygame.display.get_desktop_sizes())

        idx = self.display_index
        if idx >= num_displays:
            print(
                f"[projector_window] 警告: display_index={idx} は存在しません"
                f"（検出されたディスプレイ数={num_displays}）。display_index=0 にフォールバックします。"
            )
            idx = 0

        try:
            desktops = list(pygame.display.get_desktop_sizes())
        except Exception:
            desktops = []

        window_size = (self.width, self.height)
        flags = pygame.NOFRAME
        if self.fullscreen:
            if self.mode == "fullscreen":
                flags |= pygame.FULLSCREEN
            elif desktops:
                # 枠なしウィンドウでディスプレイを丸ごと覆う。真のフルスクリーンと違い
                # 専用 Space を作らないので、他ウィンドウをクリックしても引っ込まない。
                window_size = tuple(desktops[idx])

        try:
            screen = pygame.display.set_mode(window_size, flags, display=idx)
        except TypeError:
            # 古い pygame は display= 引数を持たないので位置引数なしにフォールバック
            screen = pygame.display.set_mode(window_size, flags)

        pygame.display.set_caption("Projection Mapping")
        pygame.mouse.set_visible(False)

        raise_it = self.above_menu_bar
        if raise_it is None:
            raise_it = self.fullscreen and self.mode == "borderless" and idx != 0
        if raise_it and _raise_above_menu_bar(pygame):
            print("[projector_window] メニューバーより前面にしました（投影面の上端に"
                  "ステータスバーが写り込みません）。")

        # 何番のどのサイズで開いたかを必ず出す。「投影されない」「位置がずれる」の
        # 原因はほぼここで、黙って別のディスプレイに開いていると気づけない。
        if desktops:
            actual = tuple(desktops[idx])
            print(f"[projector_window] ディスプレイ {len(desktops)} 台 {desktops} / "
                  f"display_index={idx} / mode={self.mode if self.fullscreen else 'windowed'} "
                  f"/ ウィンドウ {window_size[0]}x{window_size[1]}")
            if self.fullscreen and actual != (self.width, self.height):
                print(f"[projector_window] 警告: config の projector.width/height は "
                      f"{self.width}x{self.height} ですが、このディスプレイの実サイズは "
                      f"{actual[0]}x{actual[1]} です。")
                print(f"[projector_window]   描画は実サイズへ引き伸ばして表示します。"
                      f"config を {actual[0]} / {actual[1]} に直すほうが解像度を無駄にしません。")

        self._pygame = pygame
        self._screen = screen
        # 実際に確保できたサーフェスの大きさ。config の想定と違っても、描画側は
        # config 解像度のまま作り、ここで一度だけ引き伸ばして辻褄を合わせる。
        self._surface_size = screen.get_size()
        self._clock = pygame.time.Clock()
        self._opened = True

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.mouse.set_visible(True)
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._opened = False

    def __enter__(self) -> "ProjectorWindow":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _drain_events(self) -> None:
        """SDL イベントを1か所で吸い上げ、キー押下を内部バッファに貯める。

        pygame.event.get() は呼んだ側がイベントを消費してしまうため、
        終了判定とアプリのキー操作で取り合いになる。ここで一元化する。
        """
        if not self._opened:
            self._quit_requested = True
            return
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit_requested = True
                else:
                    name = pygame.key.name(event.key)
                    if name:
                        self._key_buffer.append(name)

    def pop_keys(self) -> list:
        """前回の呼び出し以降に押されたキー名（'a' 'left' 'space' など）を返して消す。"""
        self._drain_events()
        keys, self._key_buffer = self._key_buffer, []
        return keys

    def poll_quit_requested(self) -> bool:
        """ESC キーやウィンドウクローズが要求されたら True。"""
        self._drain_events()
        return self._quit_requested

    def show(self, frame_bgr: np.ndarray, fps_limit: int = 60) -> bool:
        """frame_bgr (H, W, 3) uint8 を表示する。継続する場合 True、終了要求なら False。"""
        if not self._opened:
            raise RuntimeError("open() を呼んでから show() してください。")
        if self.poll_quit_requested():
            return False

        pygame = self._pygame
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._surface_size:
            import cv2
            frame_bgr = cv2.resize(frame_bgr, self._surface_size)

        # BGR(H,W,3) -> RGB(H,W,3) -> pygame は (W,H,3) を期待するので転置する
        frame_rgb = frame_bgr[:, :, ::-1]
        surface = self._pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        self._clock.tick(fps_limit)
        return True
