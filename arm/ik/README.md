# 固定配置のIKと姿勢テーブル生成

ラックと提供先をアーム基準座標で実測し、SO-101公式URDFとPlaCoで一度だけIKを解きます。
本番時は生成済みraw tickをXIAOが再生するため、ネットワークもPC側IKも不要です。

## 1. 専用環境と公式URDF

メインアプリはNumPy 1系を使用するため、IKだけ別venvに分離します。

```bash
uv venv --python 3.12 arm/ik/.venv
uv pip install --python arm/ik/.venv/bin/python -r arm/ik/requirements.txt

git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/TheRobotStudio/SO-ARM100.git arm/ik/vendor/SO-ARM100
git -C arm/ik/vendor/SO-ARM100 sparse-checkout set Simulation/SO101
ln -sfn SO-ARM100/Simulation/SO101 arm/ik/vendor/SO101
```

## 2. 関節校正

推奨は対話式校正です。画面の指示に従って中央、各関節の負側／正側、グリッパーの
開閉位置を手で作り、各位置で Enter を押します。移動指令は送らず、接続中は常に
トルクOFFです。測定後に `direction` と安全範囲を計算し、この設定ファイルへ保存します。

```bash
uv run python arm/calibrate_xiao.py --port /dev/cu.usbmodemXXXX
```

完了後も `configured=false` のままなので、固定点の入力前にアームが動くことはありません。
可動域が0/4095をまたぐ軸を検出すると、中位補正を提案します。画面で `c` を選び、その軸を
物理的な可動範囲の中央へ戻して確認すると、STS3215の中位校正機能で現在位置を約2048 tickへ
対応付けてから、その軸だけを再測定します。中位補正値はサーボへ永続保存されます。

以下は手動で値を確認する場合の手順です。

XIAOを書き込み、最初は必ずアーム電源を切った状態でUSB通信を確認します。

```bash
uv run python arm/xiao_console.py ping --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py status --port /dev/cu.usbmodemXXXX
```

次にサーボ電源を入れ、トルクを切って手で動かせる状態にします。

```bash
uv run python arm/xiao_console.py torque-off --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py read --port /dev/cu.usbmodemXXXX
```

各関節について以下を `fixed_poses.json` に記録します。

- `zero_tick`: URDFの関節角0°に物理的に合わせたときのraw tick
- `direction`: 正方向へ動かしたときtickが増えるなら `1`、減るなら `-1`
- `min_tick` / `max_tick`: 他部品や机に衝突しない実用範囲。0/4095をそのまま使わない
- グリッパの `open_tick` / `closed_tick`: 食器を落とさず、サーボを押し続けない値

SO-101は組み立て時のホーン位置で零点が変わります。別個体のtick値はコピーしないでください。

## 3. 固定点を測る

推奨は対話式ティーチングです。トルクOFFのアームを画面で指定された8姿勢へ手で動かし、
各位置で Enter を押します。校正済みtickから関節角とTCPのXYZ/RPYを順運動学で自動計算し、
IKが同じ関節解を選べるよう実測関節角もseedとして保存します。

```bash
uv pip install --python arm/ik/.venv/bin/python -r arm/ik/requirements.txt
arm/ik/.venv/bin/python arm/teach_fixed_poses.py --port /dev/cu.usbmodemXXXX
```

一部だけ調整し直す場合は `--poses` を使います。既存の他姿勢は保持されます。

```bash
arm/ik/.venv/bin/python arm/teach_fixed_poses.py \
  --port /dev/cu.usbmodemXXXX --poses HOME RETREAT
```

食器はつかまず、グリッパーを開いたまま把持位置へ合わせます。完了後も
`configured=false`を維持するため、IK検証と再書込み前にアームが自動で動くことはありません。

以下は座標を定規で測って手入力する場合の説明です。

座標は `base_link` 原点、単位mです。ラックと提供先はベースへ剛固定し、最低限次を測ります。

- フォーク／箸それぞれの把持点 `PICK_*`
- 各把持点の真上60mm程度の `APPROACH_*`
- 客席側の `PLACE_UTENSIL`
- その真上の `APPROACH_PLACE`
- 投影から外れる `RETREAT`

`rpy_deg` はグリッパ姿勢です。SO-101本体は5自由度なので任意の6D姿勢を完全には満たせません。
位置を優先するため既定の `orientation_weight` は0.01です。

## 4. 生成と検証

値を全て入力し、最後にだけ `"configured": true` にします。

```bash
arm/ik/.venv/bin/python arm/ik/solve_fixed_poses.py --check
arm/ik/.venv/bin/python arm/ik/solve_fixed_poses.py
pio run -d arm/firmware_xiao
```

IK位置誤差が3mmを超える姿勢、校正可動範囲を超える関節角、未入力値がある設定はheaderを
生成しません。実際の再生順で1区間の単一関節変化が120度を超える場合も、急な反転を
避けるため生成を止めます。成功すると `firmware_xiao/include/generated_poses.h` が更新され、
`POSE_TABLE_CONFIGURED=true` になります。

## 5. 実機で段階確認

最初は食器なし・低いサーボ速度で、非常停止できる状態にして実施します。
まず `approach-*` だけを確認し、安全なら対応する `pick-*` へ進みます。`GOTO` は2点間の
衝突回避をしないため、離れた姿勢へ直接飛ばさないでください。
各ステップは200ms間隔で到達を確認し、初期位置からの移動量に応じて最大12秒待ちます。
期限内に到達しない場合は `DIAG joint=... current=... target=... diff=...` を表示してから
トルクを切ります。

```bash
pio run -d arm/firmware_xiao -t upload
uv run python arm/xiao_console.py home --listen 5 --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py approach-fork --listen 5 --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py pick-fork --listen 5 --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py approach-chopsticks --listen 5 --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py pick-chopsticks --listen 5 --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py approach-place --listen 5 --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py place --listen 5 --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py fork --listen 20 --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py chopsticks --listen 20 --port /dev/cu.usbmodemXXXX
uv run python arm/xiao_console.py stop --port /dev/cu.usbmodemXXXX
```

IKは衝突回避を保証しません。各区間を目視確認し、必要なら中間点を増やしてから食器を把持します。
