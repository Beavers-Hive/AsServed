"""
make_diagram.py
Hackster の記事に載せる系統図を書き出す。

図は「何が何をしているか」ではなく **どの座標系で情報が流れるか** を主題にする。
この作品の肝は、入力(指)も出力(投影)もアーム(食器)も、すべて
『机の平面のミリメートル』という一つの物差しの上で揃っていることだから。

    uv run python hackster/make_diagram.py
出力:
    hackster/images/20_system_diagram.png   1600x900
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900
BG = (12, 15, 20)
INK = (255, 255, 255)
SUB = (168, 178, 190)
ACCENT = (255, 198, 26)      # 実寸に関わる線(mm の流れ)
CYAN = (79, 200, 255)        # 機材の接続
PANEL = (24, 29, 37)
EDGE = (58, 68, 82)

FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def font(size, bold=False):
    for path in FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=1 if bold else 0)
            except Exception:
                return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def box(d, xy, title, lines, accent=EDGE, fill=PANEL):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=14, fill=fill, outline=accent, width=2)
    d.text((x0 + 22, y0 + 18), title, font=font(27, True), fill=INK)
    y = y0 + 58
    for ln in lines:
        d.text((x0 + 22, y), ln, font=font(20), fill=SUB)
        y += 28


def chip(d, xy, text, color, size=19):
    """矢印のラベル。背景を敷かないと線や箱の上で読めなくなる。"""
    f = font(size, True)
    tw = d.textlength(text, font=f)
    x, y = xy
    d.rounded_rectangle((x - tw / 2 - 10, y - size * 0.85, x + tw / 2 + 10, y + size * 0.95),
                        radius=8, fill=(10, 13, 18), outline=color, width=1)
    d.text((x - tw / 2, y - size * 0.62), text, font=f, fill=color)


def arrow(d, p0, p1, color, width=3, label=None, label_at=None, dash=False):
    x0, y0 = p0
    x1, y1 = p1
    if dash:
        n = max(2, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / 16))
        for i in range(n):
            if i % 2:
                continue
            a = (x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
            b = (x0 + (x1 - x0) * (i + 1) / n, y0 + (y1 - y0) * (i + 1) / n)
            d.line([a, b], fill=color, width=width)
    else:
        d.line([p0, p1], fill=color, width=width)
    # 矢じり
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (0.5, -0.5):
        d.line([(x1, y1),
                (x1 - 16 * math.cos(ang - s), y1 - 16 * math.sin(ang - s))],
               fill=color, width=width)
    if label:
        at = label_at or ((x0 + x1) / 2, (y0 + y1) / 2)
        chip(d, at, label, color)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((60, 46), "As Served : everything meets on one plane",
           font=font(40, True), fill=INK)
    d.text((60, 100),
           "Input, output and the robot all speak the same unit: millimetres on the table top.",
           font=font(23), fill=SUB)

    # --- 中央: 机の平面 ---------------------------------------------------
    table = (470, 330, 1130, 640)
    d.rounded_rectangle(table, radius=18, fill=(30, 37, 47), outline=ACCENT, width=3)
    d.text((500, 352), "TABLE PLANE", font=font(26, True), fill=ACCENT)
    d.text((500, 388), "table(mm) : origin and axes fixed by the printed board",
           font=font(20), fill=SUB)
    d.text((500, 424), "H_table -> proj    metric.json", font=font(20), fill=SUB)
    d.text((500, 452), "H_cam   -> proj    calibration.json", font=font(20), fill=SUB)
    d.text((500, 480), "H_cam   -> table   derived from the two above", font=font(20), fill=SUB)
    d.rounded_rectangle((500, 516, 1100, 612), radius=12, fill=(18, 23, 30), outline=EDGE)
    d.text((522, 536), "residual RMS 0.72 mm over 24 correspondences",
           font=font(22, True), fill=ACCENT)
    d.text((522, 572), "verified with a ruler, not with a screenshot",
           font=font(19), fill=SUB)

    # --- 左: 入力 ---------------------------------------------------------
    box(d, (60, 250, 400, 420), "USB camera",
        ["MediaPipe hand landmarks", "index fingertip -> cam(px)"], CYAN)
    box(d, (60, 470, 400, 640), "Printed ArUco board",
        ["6 markers, known mm layout", "one-time metric calibration"], CYAN)

    # --- 右: 出力 ---------------------------------------------------------
    box(d, (1200, 190, 1540, 360), "Ultra short throw projector",
        ["draws the dish at 1:1", "menu, size rings, cut lines"], CYAN)
    box(d, (1200, 400, 1540, 570), "XIAO ESP32-C3",
        ["half duplex TTL @ 1 Mbps", "replays taught poses"], CYAN)
    box(d, (1200, 610, 1540, 780), "SO-101 arm",
        ["fork for pasta", "chopsticks for ramen"], CYAN)

    # --- 下: 表示 ---------------------------------------------------------
    box(d, (470, 700, 1130, 830), "reTerminal E1002 : e-paper order card",
        ["the dish that was chosen on the table, shown next to it"], CYAN)

    # --- 流れ -------------------------------------------------------------
    arrow(d, (400, 320), (466, 372), ACCENT, label="cam(px) -> table(mm)", label_at=(230, 214))
    arrow(d, (400, 542), (466, 512), ACCENT, label="defines the mm frame", label_at=(230, 452))
    arrow(d, (1134, 380), (1196, 300), ACCENT, label="table(mm) -> proj(px)", label_at=(1370, 150))
    arrow(d, (1134, 486), (1196, 486), CYAN, label="BRING_UTENSIL", label_at=(1370, 381))
    arrow(d, (1370, 574), (1370, 606), CYAN, width=3)
    arrow(d, (800, 644), (800, 696), CYAN, width=3)

    d.text((60, 690), "Point and hold 3 s", font=font(26, True), fill=INK)
    for i, ln in enumerate([
            "The fingertip is mapped onto the table plane,",
            "so the hit test happens in millimetres,",
            "not in projector pixels. Targets are 70 mm",
            "or larger, sized from real pointing accuracy."]):
        d.text((60, 730 + i * 27), ln, font=font(19), fill=SUB)

    out = Path("hackster/images/20_system_diagram.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out}  {W}x{H}")


if __name__ == "__main__":
    main()
