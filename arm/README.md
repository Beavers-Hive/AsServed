# SO-101 + XIAO — 料理に合う食器を運ぶアーム

## 役割

注文された料理に応じて、**フォークまたは箸**をラックから取り、客席の固定位置へ置きます。
ピザとパスタにはフォーク、ラーメンには箸を運ぶ想定です。

アームに「意味」を持たせるのがこの構成の狙いです。メニュー選択中は動かず、注文確定時に
料理データの `utensil` を見て食器を選びます。投影UIの選択結果が、客席に届く実物へ
つながっていることが一目で分かります。

## なぜ視覚センシングを使わないか

食器はラックの固定位置にあり、置き先も固定です。**生成・確認済みの姿勢を再生するだけ**で
足ります。カメラでの把持推定は見栄えはしますが、本番デモで最も壊れやすい部分です。
19日でコンテスト動画を撮り切ることを優先し、確実に動く方を選んでいます。

SO-101公式URDFとPlaCoで固定点の逆運動学をオフライン計算し、結果をXIAOへ焼き込みます。
本番中にIKや食器検出のための視覚フィードバックループは使いません。

## 構成

| 部品 | 用途 |
|---|---|
| SO-101 (Feetech STS3215 サーボ x6) | アーム本体 |
| **Seeed Studio XIAO ESP32-C3** | サーボバスのコントローラ。ホストからのコマンドを姿勢再生に変換 |
| 食器ラック（3Dプリント） | フォークと箸を決まった位置・向きで1本ずつ保持 |
| 軽量フォーク / 箸 | アームが注文後に客席へ運ぶ実物 |

XIAO はホスト（Mac）と USB CDC シリアルで、サーボとは半二重 TTL（1Mbps）でつながります。
半二重なので、送受信の方向切り替えが要ります（レベルシフタ／方向制御付きのバスボード）。

## プロトコル

行指向のテキスト。シリアルモニタから手で叩けるので、切り分けが速い。

```
ホスト → XIAO   BRING_UTENSIL <dish_id> <fork|chopsticks>
ホスト → XIAO   HOME
ホスト → XIAO   PING
ホスト → XIAO   STATUS / READ_JOINTS
ホスト → XIAO   TORQUE_OFF / TORQUE_ON / STOP
ホスト → XIAO   GOTO <pose>（実機調整用。グリッパは開いたまま）

XIAO → ホスト   OK BRING_UTENSIL <fork|chopsticks>
XIAO → ホスト   BUSY
XIAO → ホスト   ERR <reason>
XIAO → ホスト   PONG
```

どの食器がラックのどこにあるかは XIAO 側の固定テーブルで持ちます。

ホスト側の実装は [`arm_client.py`](arm_client.py)。`--arm mock` でアーム無しでも
投影アプリの動作確認とデモのリハーサルができます。

```bash
uv run python src/table_sign.py --arm mock
uv run python src/table_sign.py --arm serial --arm-port /dev/cu.usbmodem1101
```

ポートの確認:

```bash
ls /dev/cu.usbmodem*
```

## ファームウェア

`firmware_xiao/` はPlatformIOプロジェクトです。Seeed XIAO Bus Servo Adapterの公式配線
（D7=RX、D6=TX、1Mbps）とFeetech公式Arduinoライブラリを使用します。

```bash
pio run -d arm/firmware_xiao
pio run -d arm/firmware_xiao -t upload
cd arm/firmware_xiao && pio device monitor
```

起動時は必ずトルクOFFです。`generated_poses.h` が未生成の間は
`ERR poses_not_configured` を返し、移動コマンドを拒否します。動作中も `STOP` は受け付け、
即座に軌道を中止してトルクを切ります。

## IK・ティーチング手順

1. 電源を入れ、全サーボをトルクオフにする
2. 手でアームを動かし、以下の姿勢で現在角を記録する
   - `HOME`（待機。投影の邪魔にならない位置）
   - `PICK_FORK` / `PICK_CHOPSTICKS`（ラック上の各食器の把持位置）
   - `APPROACH_*`（各把持位置の 60mm 上）
   - `PLACE_UTENSIL`（客席側の食器提供位置）
   - `RETREAT`（置いた後に投影へ写り込まない退避位置）
3. ラックと提供先の座標を実測し、PlaCo IKで関節角を求める
4. 実機で微調整した姿勢をファームウェアの姿勢テーブルに焼く
5. `BRING_UTENSIL` は APPROACH → PICK → 把持 → APPROACH → PLACE_UTENSIL → 離す → RETREAT の順で再生

フォークと箸は同じ `PLACE_UTENSIL` に置けるよう、ラック側で持ち手の把持位置を揃えると
ティーチング姿勢を共用できます。

校正・座標入力・生成の詳しい手順は [`ik/README.md`](ik/README.md) を参照してください。
実測値がまだ無いため、同梱の姿勢テーブルは安全のため未設定です。
