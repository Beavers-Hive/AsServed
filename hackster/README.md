# Hackster 投稿一式

| ファイル | 用途 |
|---|---|
| `article.md` | 記事本文（英語）。Hackster の Story にそのまま貼る |
| `elevator_pitch.md` | Elevator pitch 欄の候補と、口頭用30秒版 |
| `seeed_products_answer.md` | 応募フォームの「Seeed 製品をどう使ったか」への回答 |
| `images/00_thumbnail.gif` | **カバー画像**（アニメGIF, 800x450, 4.0秒, 3.4MB） |
| `images/*.jpg` `*.png` | 本文中の画像。記事の該当箇所へ差し込む |
| `make_diagram.py` | 系統図（`images/20_system_diagram.png`）の生成スクリプト |

## 投稿手順

1. Hackster で新規プロジェクトを作り、`article.md` の本文を Story に貼る
   （Elevator pitch 欄には `elevator_pitch.md` の本命を入れる）
2. カバー画像に `images/00_thumbnail.gif` を指定する
3. 本文中の `![...](images/xx.jpg)` の位置に、対応する画像をアップロードして差し込む
   （Hackster は相対パスを解決しないので、画像は個別にアップロードが必要）
4. Things に記事末尾の一覧を入力する
5. Video に完成動画（`video_maker/out/as_served.mp4`）の YouTube URL を貼る
6. Code に GitHub リポジトリを紐づける

## 画像の対応表

| ファイル | 内容 |
|---|---|
| `00_thumbnail.gif` | 実寸で投影された料理 → Order 長押し → 机全体がアンバーに変わるまで |
| `01_lifesize_pizza.jpg` | 実寸で投影されたマルゲリータ |
| `02_size_compare.jpg` | 選ばなかったサイズが破線で重なる |
| `03_point_and_hold.jpg` | 指で机を指している |
| `04_order_placed.jpg` | ORDER PLACED（全面アンバー） |
| `05_arm_picks_fork.jpg` | アームがフォークを掴む |
| `06_fork_delivered.jpg` | 食器が机に置かれた |
| `07_epaper_card.jpg` | reTerminal E1002 の注文カード |
| `08_projected_vs_real.jpg` | 投影と実物のピザが重なる |
| `09_setup_wide.jpg` | 機材全体の引き |
| `10_ramen.jpg` | ラーメンを実寸投影 |
| `11_ui_pizza_L.png` | 投影UIのレンダリング（実機の較正値で生成） |
| `14_calibration_board.png` | 較正ボードの印刷ページ |
| `20_system_diagram.png` | 系統図 |

再生成:

```bash
.venv/bin/python hackster/make_diagram.py
```
