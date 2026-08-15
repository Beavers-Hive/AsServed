"""親指と人差し指の距離計測を、カメラもモデルも使わずに検証する。

実機で確かめようとすると「認識」「較正」「計算」のどれが悪いのか切り分けられない。
計算とヒステリシスはここで固めておき、実機では認識と較正だけを疑えばよい状態にする。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pinch_distance as pd  # noqa: E402
from geometry import TableFrame  # noqa: E402
from pointer import INDEX_FINGER_TIP, THUMB_TIP  # noqa: E402


def landmarks(thumb, index, count: int = 21):
    """MediaPipe のランドマーク列を模した並び。使うのは2点だけ。"""
    lm = [types.SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(count)]
    lm[THUMB_TIP] = types.SimpleNamespace(x=thumb[0], y=thumb[1],
                                          z=thumb[2] if len(thumb) > 2 else 0.0)
    lm[INDEX_FINGER_TIP] = types.SimpleNamespace(x=index[0], y=index[1],
                                                 z=index[2] if len(index) > 2 else 0.0)
    return lm


# --- 距離そのもの ----------------------------------------------------------

def test_tips_px_scales_normalized_landmarks():
    tips = pd.tips_px_from_landmarks(landmarks((0.5, 0.5), (0.75, 0.5)), 1280, 720)
    assert tips.shape == (2, 2)
    np.testing.assert_allclose(tips[0], [640.0, 360.0])
    np.testing.assert_allclose(tips[1], [960.0, 360.0])


def test_world_distance_is_millimetres():
    """hand_world_landmarks はメートル。40mm 離した手は 40mm と出る。"""
    lm = landmarks((0.0, 0.0, 0.0), (0.024, 0.032, 0.0))   # 3-4-5 で 0.040 m
    assert pd.world_distance_mm(lm) == pytest.approx(40.0, abs=1e-6)


def test_table_distance_uses_calibration():
    """cam→table を通すので、カメラの解像度ではなく机の mm で出る。"""
    # 4 px = 1 mm の素直な cam→proj と、等倍の table→proj を組み合わせる
    frame = TableFrame(H_table_to_proj=np.eye(3),
                       H_cam_to_proj=np.diag([0.25, 0.25, 1.0]))
    tips_mm = pd.table_tips_mm(frame, np.array([[100.0, 200.0], [300.0, 200.0]]))
    np.testing.assert_allclose(tips_mm[0], [25.0, 50.0], atol=1e-4)
    assert pd._dist(tips_mm[0], tips_mm[1]) == pytest.approx(50.0, abs=1e-3)


# --- つまみ判定（ヒステリシス） ---------------------------------------------

def test_pinch_gate_has_hysteresis():
    gate = pd.PinchGate(close_mm=30.0, open_ratio=1.35)   # 開くのは 40.5mm
    assert gate.update(50.0) is False
    assert gate.update(35.0) is False        # 閉じる閾値には届いていない
    assert gate.update(20.0) is True
    assert gate.update(35.0) is True         # 一度つまんだら 40.5mm までは維持
    assert gate.update(45.0) is False


def test_pinch_gate_releases_when_hand_is_lost():
    gate = pd.PinchGate(close_mm=30.0)
    gate.update(10.0)
    assert gate.update(None) is False


# --- 表示に渡す値の選び方 ---------------------------------------------------

def test_value_falls_back_when_requested_source_is_missing():
    """較正が無い環境で --source table を選んでも、黙って何も出さないのではなく world を使う。"""
    s = pd.PinchSample(at=0.0, world_mm=42.0)
    assert s.value_mm("table") == pytest.approx(42.0)
    assert s.value_mm("world") == pytest.approx(42.0)

    both = pd.PinchSample(at=0.0, world_mm=42.0, table_mm=45.0)
    assert both.value_mm("table") == pytest.approx(45.0)
    assert both.value_mm("world") == pytest.approx(42.0)


def test_detail_line_reports_missing_values():
    s = pd.PinchSample(at=0.0)
    assert s.visible is False
    assert "world --" in pd.format_detail(s) and "table --" in pd.format_detail(s)


# --- 平滑化と欠測（PinchMeter を推論結果だけで動かす） -----------------------

class _Result:
    def __init__(self, hand=None, world=None):
        self.hand_landmarks = [hand] if hand else []
        self.hand_world_landmarks = [world] if world else []


def _meter(frame=None, alpha=0.5, filter_kind="ema"):
    # camera / model は measure() では使わないので None のまま組み立てる
    return pd.PinchMeter(camera=None, model_path=Path("unused.task"), frame=frame,
                         ema_alpha=alpha, filter_kind=filter_kind)


def test_measure_smooths_towards_the_new_position():
    m = _meter(alpha=0.5)
    hand = landmarks((0.5, 0.5), (0.5, 0.5))
    world = landmarks((0.0, 0.0, 0.0), (0.02, 0.0, 0.0))     # 20mm
    first = m.measure(_Result(hand, world), 100, 100, 1.0)
    assert first.world_mm == pytest.approx(20.0)

    world2 = landmarks((0.0, 0.0, 0.0), (0.04, 0.0, 0.0))    # 40mm へ跳ぶ
    second = m.measure(_Result(hand, world2), 100, 100, 1.1)
    assert second.world_mm == pytest.approx(30.0)            # alpha=0.5 の中間


def test_measure_resets_smoothing_when_the_hand_disappears():
    m = _meter(alpha=0.5)
    hand = landmarks((0.5, 0.5), (0.5, 0.5))
    world = landmarks((0.0, 0.0, 0.0), (0.02, 0.0, 0.0))
    m.measure(_Result(hand, world), 100, 100, 1.0)

    lost = m.measure(_Result(), 100, 100, 1.1)
    assert lost.visible is False and lost.world_mm is None

    # 手が戻ったら、消える前の値を引きずらず新しい値から始める
    world2 = landmarks((0.0, 0.0, 0.0), (0.04, 0.0, 0.0))
    again = m.measure(_Result(hand, world2), 100, 100, 1.2)
    assert again.world_mm == pytest.approx(40.0)


def test_measure_reports_table_mm_only_when_calibrated():
    hand = landmarks((0.25, 0.5), (0.75, 0.5))              # 幅 100px の画像で 50px 離れる
    plain = _meter().measure(_Result(hand), 100, 100, 1.0)
    assert plain.table_mm is None and plain.tips_mm is None
    assert plain.px == pytest.approx(50.0)

    frame = TableFrame(H_table_to_proj=np.eye(3), H_cam_to_proj=np.diag([0.25, 0.25, 1.0]))
    calibrated = _meter(frame=frame).measure(_Result(hand), 100, 100, 1.0)
    assert calibrated.table_mm == pytest.approx(12.5, abs=1e-3)   # 50px / 4


# --- 左右・遠近で値が変わる原因の切り分け ----------------------------------

def hand_with_scale(pinch_m: float, scale_m: float):
    """つまみ幅と『手首→中指付け根』を指定した world ランドマーク列。"""
    lm = landmarks((0.0, 0.0, 0.0), (pinch_m, 0.0, 0.0))
    lm[pd.WRIST] = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
    lm[pd.MIDDLE_FINGER_MCP] = types.SimpleNamespace(x=0.0, y=scale_m, z=0.0)
    return lm


def test_hand_scale_is_the_rigid_reference_length():
    assert pd.hand_scale_mm(hand_with_scale(0.05, 0.095)) == pytest.approx(95.0)


def test_handedness_is_reported_and_tolerates_absence():
    class R:
        handedness = [[types.SimpleNamespace(category_name="Left", score=0.98)]]
    assert pd.handedness_label(R()) == "Left"
    assert pd.handedness_label(_Result()) is None      # handedness を持たない結果でも落ちない


def test_ref_mm_cancels_a_drifting_model_scale():
    """モデルの推定スケールが左右で 10% ずれても、実測長で正規化すれば同じ値になる。"""
    left = hand_with_scale(0.050, 0.095)               # 素直に見えている側
    right = hand_with_scale(0.055, 0.1045)             # 手ごと 10% 大きく推定された側

    raw = _meter(filter_kind="none")
    a = raw.measure(_Result(landmarks((0.5, 0.5), (0.5, 0.5)), left), 100, 100, 1.0)
    raw._world_filter.reset(); raw._scale_filter.reset()
    b = raw.measure(_Result(landmarks((0.5, 0.5), (0.5, 0.5)), right), 100, 100, 2.0)
    assert b.world_mm - a.world_mm == pytest.approx(5.0, abs=0.1)     # 補正なしでは 5mm 違う

    fixed = pd.PinchMeter(camera=None, model_path=Path("unused.task"),
                          filter_kind="none", ref_mm=95.0)
    c = fixed.measure(_Result(landmarks((0.5, 0.5), (0.5, 0.5)), left), 100, 100, 1.0)
    d = fixed.measure(_Result(landmarks((0.5, 0.5), (0.5, 0.5)), right), 100, 100, 2.0)
    assert c.world_mm == pytest.approx(50.0, abs=0.1)
    assert d.world_mm == pytest.approx(50.0, abs=0.1)   # 正規化すると一致する


def test_tilt_is_zero_side_on_and_ninety_facing_the_camera():
    """つまみ軸が画像平面と平行なら 0度、カメラ軸を向いていれば 90度。"""
    side_on = landmarks((0.0, 0.0, 0.0), (0.05, 0.0, 0.0))
    facing = landmarks((0.0, 0.0, 0.0), (0.0, 0.0, 0.05))
    diagonal = landmarks((0.0, 0.0, 0.0), (0.05, 0.0, 0.05))
    assert pd.pinch_axis_tilt_deg(side_on) == pytest.approx(0.0, abs=0.1)
    assert pd.pinch_axis_tilt_deg(facing) == pytest.approx(90.0, abs=0.1)
    assert pd.pinch_axis_tilt_deg(diagonal) == pytest.approx(45.0, abs=0.1)
    assert pd.pinch_axis_tilt_deg(landmarks((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))) == 0.0


def test_tilt_warning_only_fires_when_the_value_cannot_be_trusted():
    ok = pd.PinchSample(at=0.0, world_mm=60.0, tilt_deg=20.0)
    bad = pd.PinchSample(at=0.0, world_mm=60.0, tilt_deg=70.0)
    assert pd.tilt_warning(ok) is None
    assert pd.tilt_warning(pd.PinchSample(at=0.0)) is None       # 手が無いときは黙る
    # 70度なら机への射影は約34%。ユーザーが見て納得できる数字を出す
    assert "70" in pd.tilt_warning(bad) and "34%" in pd.tilt_warning(bad)
    assert "度" in pd.tilt_warning(bad, "ja")


def test_measure_reports_the_tilt():
    m = _meter(filter_kind="none")
    facing = landmarks((0.0, 0.0, 0.0), (0.0, 0.0, 0.05))
    s = m.measure(_Result(landmarks((0.5, 0.5), (0.5, 0.5)), facing), 100, 100, 1.0)
    assert s.tilt_deg == pytest.approx(90.0, abs=0.1)
    assert s.world_mm == pytest.approx(50.0, abs=0.1)   # 3次元距離自体は向きに依らない


def test_detail_line_shows_scale_and_handedness():
    s = pd.PinchSample(at=0.0, world_mm=50.0, hand_scale_mm=95.0, handedness="Right")
    assert "scale 95.0mm" in pd.format_detail(s) and "Right" in pd.format_detail(s)


def test_sample_log_writes_one_row_per_measurement(tmp_path):
    path = tmp_path / "out.csv"
    log = pd.SampleLog(path)
    s = pd.PinchSample(at=1.0, tips_px=np.array([[100.0, 200.0], [140.0, 200.0]]),
                       world_mm=50.0, px=40.0, hand_scale_mm=95.0, handedness="Left")
    assert log.write(s) is True
    assert log.write(s) is False                       # 同じ計測は二度書かない
    assert log.write(pd.PinchSample(at=2.0)) is False  # 手が見えていないフレームは書かない
    log.close()

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].startswith("t,handedness,cam_x,cam_y")
    assert lines[1].split(",")[1:5] == ["Left", "120.0", "200.0", "50.00"]
    assert len(lines) == 2 and log.rows == 1


# --- 値が暴れないこと ------------------------------------------------------

def noisy_hold(seconds: float = 3.0, fps: float = 30.0, value_mm: float = 40.0,
               noise_mm: float = 2.0, seed: int = 0):
    """「指を止めているつもり」の入力。真値は一定で、±noise_mm の雑音だけが乗る。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    for i in range(n):
        yield i / fps, value_mm + rng.normal(0.0, noise_mm)


def test_one_euro_cuts_jitter_while_holding_still():
    f = pd.OneEuroFilter(min_cutoff=0.7, beta=0.015)
    raw, filtered = [], []
    for t, v in noisy_hold():
        raw.append(v)
        filtered.append(f(v, t))
    # 立ち上がりを除いて比べる（初回は素通しなので）。±2mm の雑音が 1/3 以下に落ちれば、
    # 続く 1mm 刻み・1.5mm 不感帯の表示は書き換わらない。
    assert np.std(filtered[30:]) < np.std(raw[30:]) / 3.0
    assert np.mean(filtered[30:]) == pytest.approx(40.0, abs=1.0)   # 偏りは持ち込まない


def test_one_euro_still_follows_a_real_pinch():
    """静止時を静かにした代償で、つまむ動き（0.3秒で 70→20mm）に追従できないと困る。"""
    f = pd.OneEuroFilter(min_cutoff=0.7, beta=0.015)
    out = 0.0
    for i in range(30):                      # 1秒静止
        out = f(70.0, i / 30.0)
    for i in range(30, 39):                  # 0.3秒でつまむ
        target = 70.0 - 50.0 * (i - 29) / 9.0
        out = f(target, i / 30.0)
    for i in range(39, 51):                  # 0.4秒保持
        out = f(20.0, i / 30.0)
    assert out == pytest.approx(20.0, abs=3.0)


def test_stable_readout_holds_the_number_against_small_wobble():
    r = pd.StableReadout(step_mm=1.0, deadband_mm=1.5)
    shown = [r.update(v) for _, v in noisy_hold(noise_mm=0.6)]
    assert len(set(shown)) == 1              # 数字は一度も書き換わらない
    assert r.text().endswith(" mm") and "." not in r.text()   # 1mm 刻みなら小数を出さない


def test_stable_readout_follows_a_real_change():
    r = pd.StableReadout(step_mm=1.0, deadband_mm=1.5)
    r.update(40.0)
    assert r.update(40.7) == pytest.approx(40.0)   # 不感帯の中は据え置き
    assert r.update(45.0) == pytest.approx(45.0)   # 本当に動いたら追従する


def test_stable_readout_forgets_the_value_when_the_hand_is_lost():
    r = pd.StableReadout()
    r.update(40.0)
    assert r.update(None) is None and r.text() == "--"


def test_filter_kinds_are_selectable():
    assert isinstance(pd.make_filter("euro"), pd.OneEuroFilter)
    assert isinstance(pd.make_filter("ema"), pd.EmaFilter)
    assert pd.make_filter("none")(12.3) == pytest.approx(12.3)
    with pytest.raises(ValueError):
        pd.make_filter("bogus")


def test_filters_pass_the_first_sample_through_and_reset():
    for kind in ("euro", "ema", "none"):
        f = pd.make_filter(kind)
        assert f(50.0, 1.0) == pytest.approx(50.0)     # 初回は素通し
        f(10.0, 1.1)
        f.reset()
        assert f(80.0, 2.0) == pytest.approx(80.0)     # reset 後も引きずらない


def test_one_euro_smooths_fingertip_arrays_too():
    """指先座標にも同じフィルタを使う。形を保ったまま平滑化されること。"""
    f = pd.OneEuroFilter()
    first = f(np.array([[0.0, 0.0], [10.0, 0.0]]), 0.0)
    second = f(np.array([[100.0, 0.0], [110.0, 0.0]]), 1.0 / 30.0)
    assert second.shape == (2, 2)
    assert 0.0 < second[0][0] < 100.0        # 跳ばずに追いかける


# --- ラベルの逃がし方 ------------------------------------------------------

def test_label_anchor_is_perpendicular_and_above():
    a = pd.label_anchor((0.0, 0.0), (100.0, 0.0), 20.0)
    np.testing.assert_allclose(a, [50.0, -20.0])        # 中点から真上へ 20

    # 線が縦でも中点から 20 離れ、線上には乗らない
    b = pd.label_anchor((0.0, 0.0), (0.0, 100.0), 20.0)
    assert pd._dist(b, (0.0, 50.0)) == pytest.approx(20.0)
    assert abs(b[0]) == pytest.approx(20.0)


def test_label_anchor_side_can_be_forced():
    """机の mm のように y の向きが表示と一致しない空間では、呼ぶ側が向きを決める。"""
    up = pd.label_anchor((0.0, 0.0), (100.0, 0.0), 20.0, side=1.0)
    down = pd.label_anchor((0.0, 0.0), (100.0, 0.0), 20.0, side=-1.0)
    assert up[1] == pytest.approx(-down[1])


def test_label_anchor_survives_fully_closed_fingers():
    """つまみきって2点が重なっても、ゼロ除算せずに線の外へ逃げる。"""
    a = pd.label_anchor((10.0, 10.0), (10.0, 10.0), 20.0)
    np.testing.assert_allclose(a, [10.0, -10.0])


# --- 描画（落ちないこと。数値は上のテストで担保済み） -----------------------

def test_annotate_camera_draws_without_a_hand():
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    out = pd.annotate_camera(canvas, pd.PinchSample(at=0.0), "--", hud="dbg")
    assert out is canvas and canvas.any()          # 少なくとも文字は描かれている


def test_draw_projection_marks_both_fingertips():
    frame = TableFrame(H_table_to_proj=np.eye(3))
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    sample = pd.PinchSample(at=0.0, tips_px=np.array([[0.0, 0.0], [1.0, 0.0]]),
                            tips_mm=np.array([[200.0, 200.0], [260.0, 200.0]]),
                            world_mm=60.0, table_mm=60.0, px=240.0)
    pd.draw_projection(canvas, frame, sample, (0.0, 0.0, 600.0, 400.0), "60 mm",
                       source="table", pinched=False, lang="en")
    # 指先の十字が両端に出ている
    assert canvas[200, 200].any() and canvas[200, 260].any()
