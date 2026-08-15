"""メニュー定義と描画が機材なしで成立するかを確かめる。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import render  # noqa: E402
from geometry import TableFrame  # noqa: E402
from menu import Menu, Portion  # noqa: E402


@pytest.fixture(scope="module")
def menu() -> Menu:
    return Menu.load(ROOT / "menu.json", project_root=ROOT)


@pytest.fixture(scope="module")
def frame() -> TableFrame:
    return TableFrame(H_table_to_proj=np.array([
        [3.0, 0.0, 200.0],
        [0.0, 3.0, 100.0],
        [0.0, 0.0005, 1.0],
    ]))


def test_shipped_menu_is_valid(menu):
    assert menu.validate() == []


def test_shipped_menu_images_exist(menu):
    for dish in menu.dishes:
        assert dish.image_path.exists(), f"{dish.id} の画像がありません: {dish.image_path}"
        assert dish.image_bgra.shape[2] == 4


def test_shipped_menu_has_fork_and_chopsticks(menu):
    utensils = {dish.utensil for dish in menu.dishes}
    assert utensils == {"fork", "chopsticks"}


def test_portions_grow_monotonically(menu):
    """S<M<L が重量でも寸法でも成り立っていること。実寸比較の説得力の前提。"""
    for dish in menu.dishes:
        grams = [p.served_g for p in dish.portions]
        food = [p.food_diameter_mm for p in dish.portions]
        plate = [p.plate_diameter_mm for p in dish.portions]
        assert grams == sorted(grams), f"{dish.id}: {grams}"
        assert food == sorted(food), f"{dish.id}: {food}"
        assert plate == sorted(plate), f"{dish.id}: {plate}"


def test_portion_lookup_is_case_insensitive(menu):
    dish = menu.dishes[0]
    assert dish.portion("l").label == "L"
    with pytest.raises(KeyError):
        dish.portion("XL")


def test_validate_catches_food_larger_than_plate(tmp_path):
    bad = {
        "dishes": [{
            "id": "bad", "name_ja": "ダメ", "name_en": "Bad",
            "image": "assets/dishes/napolitan.png",
            "portions": [
                {"label": "S", "dry_g": 80, "served_g": 180, "kcal": 480, "price_yen": 880,
                 "plate_diameter_mm": 200, "food_diameter_mm": 240},
            ],
        }]
    }
    path = tmp_path / "menu.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    problems = Menu.load(path, project_root=ROOT).validate()
    assert any("超えています" in p for p in problems)


def test_draw_image_mm_lands_where_expected(frame):
    """幅 100mm の画像が、その mm 矩形を透視変換した位置ぴったりに描かれること。

    遠近があるので「中心の倍率 x 100mm」にはならない。ホモグラフィで変換した
    4隅の外接矩形と一致することを確かめる。
    """
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    img = np.full((64, 64, 4), 255, dtype=np.uint8)

    render.draw_image_mm(canvas, frame, img, (100.0, 100.0), width_mm=100.0, height_mm=100.0)

    xs = np.where(canvas.max(axis=(0, 2)) > 0)[0]
    ys = np.where(canvas.max(axis=(1, 2)) > 0)[0]
    assert len(xs) > 0 and len(ys) > 0, "何も描かれていません"

    from geometry import rect_mm
    quad = frame.table_to_proj(rect_mm(100.0, 100.0, 100.0, 100.0))
    assert abs(xs[0] - quad[:, 0].min()) <= 2
    assert abs(xs[-1] - quad[:, 0].max()) <= 2
    assert abs(ys[0] - quad[:, 1].min()) <= 2
    assert abs(ys[-1] - quad[:, 1].max()) <= 2


def test_text_image_is_transparent_outside_glyphs():
    img = render.text_image("100 mm", 48)
    assert img.shape[2] == 4
    assert img[:, :, 3].max() == 255      # 文字がある
    assert img[:, :, 3].min() == 0        # 背景は透過


def test_draw_text_mm_returns_width_and_draws(frame):
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    width_mm = render.draw_text_mm(canvas, frame, "100 mm", (60.0, 60.0), height_mm=20.0)
    assert width_mm > 20.0
    assert canvas.max() > 0


def test_draw_text_mm_ignores_empty_string(frame):
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert render.draw_text_mm(canvas, frame, "", (10.0, 10.0), height_mm=10.0) == 0.0
    assert canvas.max() == 0


def test_draw_plate_dashes_uses_less_ink_than_solid(frame):
    solid = np.zeros((720, 1280, 3), dtype=np.uint8)
    dashed = np.zeros((720, 1280, 3), dtype=np.uint8)
    render.draw_plate_mm(solid, frame, (150.0, 120.0), 220.0, dashes=0)
    render.draw_plate_mm(dashed, frame, (150.0, 120.0), 220.0, dashes=24)
    assert 0 < int((dashed > 0).sum()) < int((solid > 0).sum())
