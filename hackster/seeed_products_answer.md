# "Briefly describe your project and how it incorporates Seeed products."

Seeed 製品の役割を**それぞれ1つの動詞**で言い切るのが要点。
「使いました」ではなく「これが無いと成立しない」と読めるようにする。

---

## 本命（約 130 words）

As Served turns a restaurant table into the menu. An ultra short throw projector draws
each dish onto the table at its **real size** — calibrated against a printed board and
verified with a ruler, at 0.72 mm residual — so guests can see how much food actually
arrives before they order. You choose a dish and a size by pointing at the table and
holding for three seconds; there is no touchscreen and nothing for a guest to touch.

Three Seeed products make the physical half work. A **XIAO ESP32-C3**, wired through the
**XIAO Bus Servo Adapter**, drives the SO-101 arm's Feetech servo bus over half-duplex
TTL at 1 Mbps and replays the taught poses that pick up the right utensil — chopsticks
for ramen, a fork for pasta. A **reTerminal E1002** stands beside the table and shows the
menu and the order card on e-paper, so the choice made in light also exists on a surface
the kitchen and the guest can read.

---

## 短い版（約 60 words / 入力欄が小さいとき）

As Served projects each dish onto the table at its real size, so guests see the portion
before they order. Pointing and holding for three seconds places the order. A Seeed
**XIAO ESP32-C3** on the **XIAO Bus Servo Adapter** drives the SO-101 arm that delivers
the matching utensil, and a **reTerminal E1002** shows the menu and order card on e-paper.

---

## 一文版（見出しやキャプション用）

A Seeed XIAO ESP32-C3 drives the arm that brings your utensil, and a reTerminal E1002
holds the order card, for a table that shows every dish at its real size.

---

## 書くときに外さない点

| 製品 | 記事での役割 | 言い方 |
|---|---|---|
| XIAO ESP32-C3 | SO-101 のサーボバス制御 | "drives the servo bus / replays taught poses" |
| XIAO Bus Servo Adapter | 半二重 TTL の配線 | "wired through" |
| reTerminal E1002 | メニューと注文カードの電子ペーパー表示 | "shows the menu and the order card" |

- **数字を1つだけ入れる。** 0.72 mm か 1 Mbps のどちらか。両方入れると散る
- **reTerminal は「表示している」までに留める。** 注文データを自動で送る実装が入ったら
  "the order is pushed to the e-paper card" に差し替える
- 主役は「実寸で見せること」。Seeed 製品は**それを物理世界へつなぐ側**として書く
