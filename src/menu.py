"""
menu.py
メニューとポーション（S/M/L）の定義。すべての寸法は mm、重量は g で持つ。

このアプリの主張は「写真ではなく実寸を見せる」ことなので、データ側も
「画像を何 mm 幅で投影するか」を必ず明示する。画像に対して倍率をかける形にすると
根拠が曖昧になり、実寸であることを検証できなくなる。

menu.json の形:
{
  "dishes": [
    {
      "id": "napolitan",
      "name_ja": "ナポリタン",
      "name_en": "Napolitan",
      "image": "assets/dishes/napolitan.png",
      "utensil": "fork",
      "allergens_ja": ["小麦", "乳"],
      "allergens_en": ["wheat", "milk"],
      "portions": [
        {"label": "S", "dry_g": 80, "served_g": 180, "kcal": 480, "price_yen": 880,
         "plate_diameter_mm": 220, "food_diameter_mm": 150},
        ...
      ]
    }
  ]
}
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class Portion:
    label: str                 # "S" / "M" / "L"
    served_g: int              # 提供時の重量
    kcal: int
    price_yen: int
    plate_diameter_mm: float   # 実際に使う皿の直径
    food_diameter_mm: float    # 料理そのものの直径 = 投影画像の幅に相当する mm
    dry_g: Optional[int] = None   # 乾麺の量。パスタ等だけで使う（ピザでは意味がない）

    @staticmethod
    def from_dict(d: dict) -> "Portion":
        return Portion(
            label=str(d["label"]),
            served_g=int(d["served_g"]),
            kcal=int(d["kcal"]),
            price_yen=int(d["price_yen"]),
            plate_diameter_mm=float(d["plate_diameter_mm"]),
            food_diameter_mm=float(d["food_diameter_mm"]),
            dry_g=int(d["dry_g"]) if d.get("dry_g") is not None else None,
        )


@dataclasses.dataclass
class Dish:
    id: str
    name_ja: str
    name_en: str
    image_path: Path
    portions: list
    # 注文確定後にロボットアームが客席へ運ぶ食器。
    # 現在のラックはフォークと箸の2レーンを想定する。
    utensil: str = "fork"
    allergens_ja: list = dataclasses.field(default_factory=list)
    allergens_en: list = dataclasses.field(default_factory=list)
    # カットできる料理だけ指定する（ピザなど）。空ならカット UI もカット線も出ない。
    slice_options: list = dataclasses.field(default_factory=list)
    _image_bgra: Optional[np.ndarray] = dataclasses.field(default=None, repr=False)

    @property
    def sliceable(self) -> bool:
        return len(self.slice_options) > 0

    def portion(self, label: str) -> Portion:
        for p in self.portions:
            if p.label.upper() == label.upper():
                return p
        raise KeyError(f"{self.id} にポーション {label!r} がありません")

    @property
    def image_bgra(self) -> np.ndarray:
        if self._image_bgra is None:
            self._image_bgra = load_rgba(self.image_path)
        return self._image_bgra


def load_rgba(path: Path) -> np.ndarray:
    """料理画像を BGRA で読む。アルファが無い画像は不透明として扱う。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"料理画像を読み込めません: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.dstack([img, alpha])
    if img.shape[2] == 4:
        return img
    raise ValueError(f"未対応のチャンネル数です: {path} shape={img.shape}")


@dataclasses.dataclass
class Menu:
    dishes: list

    def __post_init__(self):
        if not self.dishes:
            raise ValueError("メニューが空です")

    def by_id(self, dish_id: str) -> Dish:
        for d in self.dishes:
            if d.id == dish_id:
                return d
        raise KeyError(f"料理 {dish_id!r} が見つかりません")

    def index_of(self, dish_id: str) -> int:
        return [d.id for d in self.dishes].index(dish_id)

    @staticmethod
    def load(path: Path, project_root: Optional[Path] = None) -> "Menu":
        path = Path(path)
        root = project_root or path.parent
        raw = json.loads(path.read_text(encoding="utf-8"))

        dishes = []
        for d in raw["dishes"]:
            image = Path(d["image"])
            if not image.is_absolute():
                image = root / image
            portions = [Portion.from_dict(p) for p in d["portions"]]
            if not portions:
                raise ValueError(f"{d['id']} にポーションがありません")
            dishes.append(Dish(
                id=str(d["id"]),
                name_ja=str(d.get("name_ja", d["id"])),
                name_en=str(d.get("name_en", d["id"])),
                image_path=image.resolve(),
                portions=portions,
                utensil=str(d.get("utensil", "fork")),
                allergens_ja=list(d.get("allergens_ja", [])),
                allergens_en=list(d.get("allergens_en", [])),
                slice_options=[int(n) for n in d.get("slice_options", [])],
            ))
        return Menu(dishes=dishes)

    def validate(self) -> list:
        """起動前チェック。画像の欠損とデータの矛盾を洗い出して警告文のリストで返す。"""
        problems = []
        for d in self.dishes:
            if d.utensil not in ("fork", "chopsticks"):
                problems.append(
                    f"{d.id}: utensil は fork / chopsticks のどちらかです ({d.utensil!r})"
                )
            if not d.image_path.exists():
                problems.append(f"{d.id}: 画像がありません → {d.image_path}")
            labels = [p.label for p in d.portions]
            if len(set(labels)) != len(labels):
                problems.append(f"{d.id}: ポーションのラベルが重複しています {labels}")
            for p in d.portions:
                if p.food_diameter_mm > p.plate_diameter_mm:
                    problems.append(
                        f"{d.id}/{p.label}: 盛り付け径 {p.food_diameter_mm:g}mm が "
                        f"皿の直径 {p.plate_diameter_mm:g}mm を超えています"
                    )
            grams = [p.served_g for p in d.portions]
            if grams != sorted(grams):
                problems.append(f"{d.id}: ポーションが重量順に並んでいません {labels}={grams}")
            for n in d.slice_options:
                if n < 2:
                    problems.append(f"{d.id}: slice_options に 2 未満の値があります ({n})")
            if d.slice_options != sorted(d.slice_options):
                problems.append(f"{d.id}: slice_options が昇順ではありません {d.slice_options}")
        return problems
