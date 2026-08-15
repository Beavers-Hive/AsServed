"""指差し選択（的の当たり判定・3秒滞留・自動レイアウト）を機材なしで検証する。

指差しUIは実機で試すと調整に時間がかかるうえ、失敗しても原因が
「認識」「較正」「ロジック」のどれか分からなくなる。ロジックはここで固めておく。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ui  # noqa: E402
from menu import Menu  # noqa: E402
from ui import _aabb  # noqa: E402  レイアウト検査は本体と同じ外接矩形定義を使う
from pointer import NullPointer, ScriptedPointer  # noqa: E402

BOUNDS = (0.0, 0.0, 900.0, 506.0)


@pytest.fixture(scope="module")
def menu() -> Menu:
    return Menu.load(ROOT / "menu.json", project_root=ROOT)


@pytest.fixture
def targets(menu):
    return ui.build_targets(menu, 0, BOUNDS, [450.0, 229.0])


# --- 的の当たり判定 --------------------------------------------------------

def test_rect_target_contains():
    t = ui.Target(id="t", kind="dish", center_mm=(100.0, 100.0), label="x",
                  shape="rect", width_mm=100.0, height_mm=80.0)
    assert t.contains((100.0, 100.0))
    assert t.contains((149.0, 139.0))
    assert not t.contains((151.0, 100.0))
    assert not t.contains((100.0, 141.0))


def test_circle_target_contains():
    t = ui.Target(id="t", kind="size", center_mm=(200.0, 150.0), label="M",
                  shape="circle", diameter_mm=90.0)
    assert t.contains((200.0, 150.0))
    assert t.contains((200.0 + 44.0, 150.0))
    assert not t.contains((200.0 + 46.0, 150.0))
    # 角は円の外（矩形判定になっていないことの確認）
    assert not t.contains((200.0 + 40.0, 150.0 + 40.0))


def test_perimeter_is_closed_and_around_target():
    for t in (ui.Target(id="a", kind="size", center_mm=(0.0, 0.0), label="M",
                        shape="circle", diameter_mm=90.0),
              ui.Target(id="b", kind="dish", center_mm=(0.0, 0.0), label="x",
                        shape="rect", width_mm=100.0, height_mm=100.0)):
        pts = t.perimeter_mm(inflate_mm=7.0)
        assert len(pts) >= 100
        r = np.linalg.norm(pts, axis=1)
        assert r.min() > 40.0 and r.max() < 110.0


# --- 自動レイアウト --------------------------------------------------------

def test_layout_targets_are_large_enough_to_point_at(targets):
    for t in targets:
        small = t.diameter_mm if t.shape == "circle" else min(t.width_mm, t.height_mm)
        assert small >= ui.MIN_TARGET_MM * 0.85, f"{t.id} が小さすぎます: {small:.0f}mm"


def test_layout_targets_stay_inside_projection_area(targets):
    x0, y0, x1, y1 = BOUNDS
    for t in targets:
        half_w = t.diameter_mm * 0.5 if t.shape == "circle" else t.width_mm * 0.5
        half_h = t.diameter_mm * 0.5 if t.shape == "circle" else t.height_mm * 0.5
        assert x0 <= t.center_mm[0] - half_w and t.center_mm[0] + half_w <= x1, t.id
        assert y0 <= t.center_mm[1] - half_h and t.center_mm[1] + half_h <= y1, t.id


def test_layout_targets_do_not_overlap(targets):
    """的が重なると、どちらを指しているのか判定が入れ替わって使い物にならない。"""
    for i, a in enumerate(targets):
        for b in targets[i + 1:]:
            ax0, ay0, ax1, ay1 = _aabb(a)
            bx0, by0, bx1, by1 = _aabb(b)
            overlap = not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)
            assert not overlap, f"{a.id} と {b.id} が重なっています"


def test_layout_has_one_target_per_dish_and_portion(menu, targets):
    kinds = [t.kind for t in targets]
    assert kinds.count("dish") == len(menu.dishes)
    assert kinds.count("size") == len(menu.dishes[0].portions)
    assert kinds.count("order") == 1


def test_layout_size_buttons_follow_selected_dish(menu):
    a = ui.build_targets(menu, 0, BOUNDS, [450.0, 229.0])
    b = ui.build_targets(menu, 1, BOUNDS, [450.0, 229.0])
    labels_a = [t.payload.label for t in a if t.kind == "size"]
    labels_b = [t.payload.label for t in b if t.kind == "size"]
    assert labels_a == labels_b == ["S", "M", "L"]
    assert [t.payload for t in a if t.kind == "size"] != [t.payload for t in b if t.kind == "size"]


def test_order_button_never_covers_the_life_size_preview(menu):
    """注文ボタンが実寸プレビューに重ならないこと。

    机が狭いほど皿が下端に迫るので、下中央へ機械的に置くと皿やアレルゲン表示を
    踏む。実機の投影範囲（内接矩形 747x393mm）を含む複数の広さで確認する。
    """
    biggest = max(p.plate_diameter_mm for d in menu.dishes for p in d.portions)
    areas = [
        ((0.0, 0.0, 900.0, 506.0), [450.0, 229.0]),
        ((-203.0, -211.0, 544.0, 181.0), [139.0, -34.0]),   # 実機の投影範囲（内接矩形）
        ((0.0, 0.0, 700.0, 420.0), [350.0, 190.0]),
    ]
    for bounds, center in areas:
        targets = ui.build_targets(menu, 0, bounds, center)
        order = next(t for t in targets if t.kind == "order")
        rect = _aabb(order)
        assert not ui._rect_hits_disc(rect, center, biggest * 0.5), \
            f"{bounds}: 注文ボタンが皿({biggest:g}mm)に重なっています @ {order.center_mm}"


# --- カット数（サイズ選択の脇役であること） ---------------------------------

def test_slice_targets_exist_for_sliceable_dishes(menu, targets):
    dish = menu.dishes[0]
    assert dish.sliceable
    slices = [t for t in targets if t.kind == "slice"]
    assert [t.payload for t in slices] == dish.slice_options


def test_slice_buttons_are_smaller_than_size_buttons(targets):
    """序列を目視ではなくデータで担保する。カット数がサイズより大きくなったら失敗させる。"""
    size_d = min(t.diameter_mm for t in targets if t.kind == "size")
    slice_d = max(t.diameter_mm for t in targets if t.kind == "slice")
    assert slice_d < size_d, f"カット数({slice_d:g}mm)がサイズ({size_d:g}mm)以上になっています"


def test_slice_buttons_are_still_pointable(targets):
    for t in targets:
        if t.kind == "slice":
            assert t.diameter_mm >= ui.MIN_TARGET_MM


def test_slice_buttons_sit_inboard_of_size_buttons(targets):
    """サイズは一番外側。手が自然に伸びる位置を主役に取らせる。"""
    size_x = min(t.center_mm[0] for t in targets if t.kind == "size")
    slice_x = max(t.center_mm[0] for t in targets if t.kind == "slice")
    assert slice_x < size_x


def test_no_slice_targets_for_a_dish_without_slice_options(tmp_path):
    raw = json.loads((ROOT / "menu.json").read_text(encoding="utf-8"))
    for d in raw["dishes"]:
        d.pop("slice_options", None)
    path = tmp_path / "menu.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    plain = Menu.load(path, project_root=ROOT)
    assert not plain.dishes[0].sliceable
    targets = ui.build_targets(plain, 0, BOUNDS, [450.0, 229.0])
    assert [t for t in targets if t.kind == "slice"] == []


def test_preview_span_leaves_room_for_the_side_columns(menu):
    """皿を置ける横幅は、左右の的の列を引いた残りであること。

    投影範囲の真ん中に皿を置くと、皿が大きいときにサイズ列やカット列と取り合う。
    """
    bounds = (-203.0, -211.0, 544.0, 181.0)
    left, right = ui.preview_span_mm(menu, bounds, {})
    assert bounds[0] < left < right < bounds[2]

    center = [(left + right) * 0.5, -34.0]
    targets = ui.build_targets(menu, 0, bounds, center, {})
    biggest = max(p.plate_diameter_mm for d in menu.dishes for p in d.portions)
    for t in targets:
        assert not ui._rect_hits_disc(_aabb(t), center, biggest * 0.5), \
            f"{t.id} が実寸プレビュー(φ{biggest:g}mm)に重なっています"


def test_layout_degrades_gracefully_when_the_area_is_too_small(menu):
    """皿を置く余地すら無い狭さでも、的の構成は壊さずに返すこと。

    この場合は重なりを避けようがない。table_sign --check が「皿が投影範囲に
    収まりません」と警告するので、レイアウト側は落ちずに返すのが正しい振る舞い。
    """
    tiny = (0.0, 0.0, 560.0, 340.0)
    targets = ui.build_targets(menu, 0, tiny, [280.0, 155.0])
    kinds = [t.kind for t in targets]
    assert kinds.count("order") == 1
    assert kinds.count("dish") == len(menu.dishes)
    for t in targets:
        x0, y0, x1, y1 = _aabb(t)
        assert tiny[0] <= x0 and x1 <= tiny[2] and tiny[1] <= y0 and y1 <= tiny[3], t.id


def test_layout_adapts_to_narrow_area(menu):
    """狭い投影範囲でも的が範囲外に出ないこと。"""
    narrow = (0.0, 0.0, 520.0, 300.0)
    for t in ui.build_targets(menu, 0, narrow, [260.0, 140.0]):
        x0, y0, x1, y1 = _aabb(t)
        assert narrow[0] <= x0 and x1 <= narrow[2], f"{t.id} が横にはみ出しています"
        assert narrow[1] <= y0 and y1 <= narrow[3], f"{t.id} が縦にはみ出しています"


# --- 3秒滞留 --------------------------------------------------------------

@pytest.fixture
def one_target():
    return [ui.Target(id="size:M", kind="size", center_mm=(100.0, 100.0), label="M",
                      shape="circle", diameter_mm=90.0),
            ui.Target(id="size:L", kind="size", center_mm=(300.0, 100.0), label="L",
                      shape="circle", diameter_mm=90.0)]


def test_dwell_fires_after_the_configured_time(one_target):
    sel = ui.DwellSelector(dwell_seconds=3.0)
    assert sel.update((100.0, 100.0), one_target, now=0.0) is None
    assert sel.update((100.0, 100.0), one_target, now=2.9) is None
    event = sel.update((100.0, 100.0), one_target, now=3.0)
    assert event is not None and event.target.id == "size:M"


def test_dwell_progress_grows_then_resets_on_fire(one_target):
    sel = ui.DwellSelector(dwell_seconds=3.0)
    sel.update((100.0, 100.0), one_target, now=0.0)
    sel.update((100.0, 100.0), one_target, now=1.5)
    assert 0.45 < sel.progress < 0.55
    sel.update((100.0, 100.0), one_target, now=3.0)
    assert sel.progress == 0.0   # 確定済みはラッチされ、進捗表示は消える


def test_dwell_does_not_refire_while_finger_stays(one_target):
    sel = ui.DwellSelector(dwell_seconds=3.0)
    sel.update((100.0, 100.0), one_target, now=0.0)
    assert sel.update((100.0, 100.0), one_target, now=3.0) is not None
    for t in (3.5, 6.5, 9.9, 20.0):
        assert sel.update((100.0, 100.0), one_target, now=t) is None


def test_dwell_refires_after_leaving_and_returning(one_target):
    sel = ui.DwellSelector(dwell_seconds=3.0, grace_seconds=0.35)
    sel.update((100.0, 100.0), one_target, now=0.0)
    assert sel.update((100.0, 100.0), one_target, now=3.0) is not None
    sel.update(None, one_target, now=3.5)      # grace を超えて離れる
    sel.update((100.0, 100.0), one_target, now=4.0)
    assert sel.update((100.0, 100.0), one_target, now=7.0) is not None


def test_dwell_survives_short_detection_dropouts(one_target):
    """MediaPipe が数フレーム落ちても滞留を捨てない。ここを厳しくすると実用にならない。"""
    sel = ui.DwellSelector(dwell_seconds=3.0, grace_seconds=0.35)
    sel.update((100.0, 100.0), one_target, now=0.0)
    sel.update((100.0, 100.0), one_target, now=1.0)
    sel.update(None, one_target, now=1.1)      # 検出ロス（grace 内）
    sel.update(None, one_target, now=1.3)
    sel.update((100.0, 100.0), one_target, now=1.4)
    assert sel.hover is not None and sel.hover.id == "size:M"
    assert sel.update((100.0, 100.0), one_target, now=3.0) is not None


def test_dwell_resets_when_moving_to_another_target(one_target):
    sel = ui.DwellSelector(dwell_seconds=3.0)
    sel.update((100.0, 100.0), one_target, now=0.0)
    sel.update((100.0, 100.0), one_target, now=2.5)
    sel.update((300.0, 100.0), one_target, now=2.6)          # L へ移動
    assert sel.update((300.0, 100.0), one_target, now=4.0) is None   # 1.4秒しか経っていない
    assert sel.update((300.0, 100.0), one_target, now=5.7) is not None


def test_dwell_ignores_points_outside_every_target(one_target):
    sel = ui.DwellSelector(dwell_seconds=3.0)
    for t in np.arange(0.0, 10.0, 0.1):
        assert sel.update((700.0, 400.0), one_target, now=float(t)) is None
    assert sel.progress == 0.0


def test_dwell_handles_no_pointer_at_all(one_target):
    sel = ui.DwellSelector(dwell_seconds=3.0)
    for t in np.arange(0.0, 5.0, 0.1):
        assert sel.update(None, one_target, now=float(t)) is None


# --- ポインタ源 ------------------------------------------------------------

def test_null_pointer_reports_nothing():
    assert NullPointer().read().point_mm is None


def test_scripted_pointer_interpolates_and_loops():
    p = ScriptedPointer([(0, 0, 0), (2, 100, 50)], loop=True)
    assert np.allclose(p.at_time(0.0).point_mm, [0, 0])
    assert np.allclose(p.at_time(1.0).point_mm, [50, 25])
    assert np.allclose(p.at_time(2.0).point_mm, [0, 0])   # 一周して先頭へ
    assert np.allclose(p.at_time(3.0).point_mm, [50, 25])


def test_scripted_pointer_rejects_empty():
    with pytest.raises(ValueError):
        ScriptedPointer([])


# --- 通し: 指差しだけで料理とサイズを選び、注文まで行けるか -------------------

def test_slice_state_defaults_and_cycling(menu):
    from table_sign import SignState  # noqa: E402

    state = SignState(menu, None, "M")
    opts = state.dish.slice_options
    assert state.slice_count == opts[-1]          # 既定は一番細かいカット

    assert state.set_slices(opts[0]) is True
    assert state.slice_count == opts[0]
    assert state.set_slices(opts[0]) is False     # 同じ値なら変化なし
    assert state.set_slices(999) is False         # 選択肢に無い値は拒否

    state.cycle_slices(+1)
    assert state.slice_count == opts[1]
    state.set_slices(opts[-1])
    state.cycle_slices(+1)
    assert state.slice_count == opts[0]           # 一周する


@pytest.fixture
def two_sliceable_menu(tmp_path) -> Menu:
    """カットできる料理が2品だけのメニュー。

    同梱メニューはカットできる料理が1品しかないので、料理をまたぐ挙動を確かめるには
    自前で用意する。こうしておけば、今後メニューを入れ替えてもこのテストは壊れない。
    """
    raw = json.loads((ROOT / "menu.json").read_text(encoding="utf-8"))
    pizza = next(d for d in raw["dishes"] if d.get("slice_options"))
    twin = json.loads(json.dumps(pizza))
    twin["id"] = pizza["id"] + "_twin"
    raw["dishes"] = [pizza, twin]
    path = tmp_path / "menu.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return Menu.load(path, project_root=ROOT)


def test_slice_choice_survives_switching_dishes(two_sliceable_menu):
    from table_sign import SignState  # noqa: E402

    state = SignState(two_sliceable_menu, None, "M")
    state.set_slices(state.dish.slice_options[0])
    chosen = state.slice_count
    state.set_dish(1)
    assert state.dish.sliceable
    assert state.slice_count == chosen


def test_slice_choice_returns_after_visiting_a_non_sliceable_dish(menu):
    """カットできない料理へ移ると 0、戻ると元の選択に復帰すること。

    メニューにピザと麺類が混在するので、切り替えのたびに好みが飛ぶと使いにくい。
    """
    from table_sign import SignState  # noqa: E402

    sliceable = [i for i, d in enumerate(menu.dishes) if d.sliceable]
    plain = [i for i, d in enumerate(menu.dishes) if not d.sliceable]
    if not sliceable or not plain:
        pytest.skip("メニューにカット可・不可の両方が必要")

    state = SignState(menu, menu.dishes[sliceable[0]].id, "M")
    chosen = state.dish.slice_options[0]
    state.set_slices(chosen)

    state.set_dish(plain[0])
    assert state.slice_count == 0                 # 麺を「4カット」と言わない

    state.set_dish(sliceable[0])
    assert state.slice_count == chosen


def test_slice_selection_does_not_move_the_arm(menu, targets):
    """注文前のカット数変更では、アームへ指示を出してはいけない。"""
    sys.path.insert(0, str(ROOT / "arm"))
    from arm_client import MockArmClient  # noqa: E402
    from table_sign import SignState, apply_selection  # noqa: E402

    state = SignState(menu, None, "M")
    arm = MockArmClient()
    target = next(t for t in targets if t.kind == "slice"
                  and t.payload != state.slice_count)
    apply_selection(ui.DwellEvent(target=target, at=0.0), state, arm)

    assert state.slice_count == target.payload
    assert arm.log == []


def test_end_to_end_selection_by_pointing(menu, targets):
    """擬似ポインタで「2品目 → 一番大きいサイズ → 注文する」をなぞる。

    メニューを差し替えても壊れないよう、料理名は menu.json から引く。
    """
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "arm"))
    from arm_client import MockArmClient  # noqa: E402
    from table_sign import SignState, apply_selection  # noqa: E402

    state = SignState(menu, None, "S")
    arm = MockArmClient()
    sel = ui.DwellSelector(dwell_seconds=3.0)

    def point_at(target_id: str, t0: float) -> float:
        target = next(t for t in current() if t.id == target_id)
        for t in np.arange(t0, t0 + 3.1, 0.1):
            event = sel.update(target.center_mm, current(), now=float(t))
            if event is not None:
                apply_selection(event, state, arm)
                return float(t) + 0.5
        raise AssertionError(f"{target_id} が確定しませんでした")

    def current():
        return ui.build_targets(menu, state.dish_index, BOUNDS, [450.0, 229.0])

    first, second = menu.dishes[0], menu.dishes[1]
    small, large = second.portions[0], second.portions[-1]
    assert state.dish.id == first.id and state.portion.label == small.label

    t = point_at(f"dish:{second.id}", 0.0)
    assert state.dish.id == second.id

    t = point_at(f"size:{large.label}", t)
    assert state.portion.label == large.label

    assert not state.ordered
    point_at("order", t)
    assert state.ordered

    # 選択中は動かず、注文確定時にだけ料理に合う食器の指示が飛ぶこと
    assert arm.log == [f"BRING_UTENSIL {second.id} {second.utensil}"]
