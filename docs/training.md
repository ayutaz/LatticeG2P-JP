# 学習データと学習

ニューラルスコアラの学習データの作り方、学習、ONNX への書き出し。
落とし穴の ID（R-xx）は [pitfalls.md](pitfalls.md) を参照。

★**学習済みモデルと学習データはリポジトリに含めていない。** 学習データは LLM で生成したものなので、
同じ手順でも同じものはできない。ここに書いた手順で作り直すことになる。

★ニューラルを使わない構成（辞書 + 連接行列）は学習が要らない。精度はそちらの方が高かった（→ [results.md](results.md)）。

## 目次

1. [全体像](#1-全体像)
2. [LLM 生成データ（多音語）](#2-llm-生成データ多音語)
3. [フィルタ](#3-フィルタ)
4. [数詞 + 助数詞](#4-数詞--助数詞)
5. [青空文庫のルビ](#5-青空文庫のルビ)
6. [train / dev の分割](#6-train--dev-の分割)
7. [学習](#7-学習)
8. [ONNX への書き出し](#8-onnx-への書き出し)
9. [クラウド GPU（vast.ai）](#9-クラウド-gpuvastai)
10. [データ形式と再現性](#10-データ形式と再現性)

---

## 1. 全体像

教師は「文」と「文全体の正しい読み」だけ（形態素分割は要らない）。データ源は 3 つある。

| データ源 | 量（このリポジトリの実績） | 読みの範囲 | 特徴 |
|---|---|---|---|
| LLM 生成（多音語） | 6,921 文 / 対象語 1,277 種類 | 文全体 | 多音語を狙って選べる。文脈的な誤読が残りうる |
| 数詞 + 助数詞 | 上に含む | 文全体 | 読みを表で確定させてから文だけ書かせる |
| 青空文庫のルビ | 11,057 文 / 対象語 7,562 種類 | 文の一部 | 人間が付けた読み。無料。**学習専用** |

分割は train 6,554 文 / dev 367 文。論文の 234 万文の 0.3% だが、データを 4 倍・語彙を 8 倍にしても
対象語は +0.95 pt しか動かなかった（[R-17](pitfalls.md#r-17)）。

```
多音語の選定 → 生成（サブエージェント / API）→ フィルタ → train / dev 分割 → 学習 → ONNX 化 → 評価
```

---

## 2. LLM 生成データ（多音語）

同表記異読（`米を炊く / 米国政府`・`生ビール / 一生`）は辞書だけでは決まらず、文脈が要る。ここに学習データを集中させる。

### 2-1. 多音語の選定

```bash
uv run python scripts/select_polyphonic.py     # → data/polyphonic.jsonl（28,091 語）
```

- 使える品詞のエントリだけを見る（固有名詞の読みを 1 つでも持つ表記を丸ごと除外しない）
- 読み数で除外せず、コストの小さい順に上位 N 個を採る（`生` は 46 読み）
- `phonetic_key` で長音の書き分けを畳んでから読み数を数える（`コウ / コー` を多音語と数えない）
- 頻度順（コスト順）に並べる。`--limit` が意味を持つように

★**件数では中身の妥当性は分からない。上位 30 件を目視する**（[R-16](pitfalls.md#r-16)）。

### 2-2. 生成

生成の経路は 2 つあり、**フィルタ以降はまったく同じ**。

**Claude Code のサブエージェント経路（既定・API キー不要）**

```bash
uv run python scripts/make_batches.py --limit 200 --batch-size 20    # → data/batches/batch_NNN.jsonl
# 各 batch_NNN.jsonl について、prompts/subagent-generate-v1.md を渡してサブエージェントを起動し、
# data/batches/batch_NNN.out.jsonl を書かせる（読み候補の検証と例文生成を 1 回で行う）
uv run python scripts/collect_batches.py                             # 回収 → フィルタ → data/train.jsonl
```

**Claude API 経路**（`.env` に `ANTHROPIC_API_KEY`）

```bash
uv run python scripts/generate_data.py --limit 200 --n-per-group 10
uv run python scripts/filter_data.py
```

サブエージェントの出力は壊れうる前提で扱う。形式が不正な行は捨て、対象語の位置も表記から復元する。
**件数はエージェントの報告ではなくファイルから数える**（報告と実ファイルが食い違ったことがある）。

### 2-3. 生成モデルの選び方

生成の品質はフィルタが機械的に判定するので、「安いモデルで足りるか」は推測ではなく実測で決められる。

1. まず安いモデル（haiku）で 1 バッチだけ試す
2. 歩留まり（ラティス整合と最終歩留まり。`collect_batches.py` が出す）を測る
3. ゲート（ラティス整合 > 95%・最終歩留まり > 85%）を割ったら 1 段上げる

★モデルで品質は大きく違う。助詞「は→ワ」の規則違反は opus 0.5% / haiku 30.6% だった。
プロンプトファイルに書いてあっても安いモデルは見落とすので、**起動時の指示で最重要ルールを再掲する**（4.4% に下がった）。

★**検証手段を渡せるなら渡す。** Hard Set の生成では、検証関数を import して自己検証するよう指示したら採用率が 50% → 100% になった。

★**安いモデルが作ったラベルを監査するときは、1 段上のモデルを使う**（同じモデルでは同じ誤りを見逃す。[R-19](pitfalls.md#r-19)）。

---

## 3. フィルタ

LLM の出力をそのまま学習に使わない（`src/lattice_g2p/data/filter.py`）。

| 段 | 内容 |
|---|---|
| 1. 形式 | 読みがカタカナのみか、対象語が文中にあるか、文長、数字・英字が混じっていないか |
| 2. ★ラティス整合 | 文と読みからラティスを作り、**正解経路集合 G が空でないか**（= Oracle と同じ処理） |
| 3. 対象語の読み | 正解経路上で対象語が意図した読みになっているか（境界を持つ経路 → 覆うノードのカナ範囲の 2 段階） |
| 4. 重複 | 完全重複と近似重複 |

加えて、助詞の読みのチェック（すべての経路で避けられない「助詞 ハ」があるときだけ落とす）と、
同じ語への矛盾ラベルの検出（`data/consistency.py`）がある。

★**ラティス整合は品質管理ではなく動作要件である。** G が空の例は CRF の分子が計算できず、損失が −∞ になる。
落ちた例は捨てるだけでなく集計する。落ちる率が高いときは、LLM ではなく辞書かコード（正規化）の問題であることが多い。

★ラティス整合は「その読みで生成できるか」しか見ないので、文脈として自然かは検出できない（[R-23](pitfalls.md#r-23)）。

---

## 4. 数詞 + 助数詞

`一本`・`三匹`・`八階` のような例が学習データに 1 つも無いと、数詞の読みは直らない（データ量や事前学習では 1 件も減らなかった）。

★**ここではラティス整合フィルタが効かない。**

```
三本 → サンホン    誤りだが 三(サン) + 本(ホン) で生成できてしまう
三本 → サンボン    正しい
```

`本` には `ホン`・`ボン`・`ポン` がすべて辞書にあるので、「生成できる」と「正しい」が一致しない。
そこで**知識（読みの表）と量（文）を分ける**。

| | 担当 | 量 |
|---|---|---|
| 読みの表（知識） | 上位のモデルが作り、人が目視する | 数詞 10 × 助数詞 30 程度 |
| 例文（量） | サブエージェント | 1 組あたり数文 |

読みを先に確定させ、**文を書くエージェントに判断の余地を与えない**。表を作らせるときは「自信の無いものを申告させる」と確認が楽になる。

```bash
uv run python scripts/make_numeral_batches.py --table data/num_table_agent.jsonl
# 各バッチについて prompts/subagent-numeral-v1.md を渡してサブエージェントに文だけ書かせる
uv run python scripts/collect_batches.py --batch-dir data/batches data/batches_num
```

★**促音化するか（`六本 → ロッポン`）は規則ではなく語彙的知識なので、数詞を dev に取り分けない**（[R-21](pitfalls.md#r-21)）。

---

## 5. 青空文庫のルビ

```
彼は｜生真面目《きまじめ》な男だった
```

実際の出版物に人間が付けた読みで、hallucination が無く、無料で、語彙が広い。
ただし読みは文の一部にしか無いので、CRF は部分制約の積グラフで学習する（[architecture.md](architecture.md#5-6-部分制約文の一部だけ読みが分かる場合)）。

```bash
uv run python scripts/fetch_aozora.py --limit 400     # 新字新仮名・著作権なしの作品だけ。作家ごとに上限
uv run python scripts/build_ruby_data.py              # 本文抽出 → ルビ抽出 → ラティス整合 → data/ruby.jsonl
```

- ★**新字新仮名に限る。** 旧仮名の作品はルビが `ケフ` / `ヰ` / `ヱ` になり、辞書の読みと一致せず全部落ちる
- 義訓（`卓子 → テーブル`）や辞書に無い固有名詞は、**ラティス整合が自動的に落とす**。辞書制約がそのまま品質フィルタになる（歩留まり 67%）
- 落とすのは span 単位。1 つのルビが使えなくても、同じ文の他のルビは使える
- ★**dev の語のルビは自動で除外される**（`drop_held_out_spans`。除外しないとルビ文の 1.5% が dev の語を対象にしていた）
- ルビは**学習専用**。文全体の読みが無いので評価には使えない

★**混ぜるより「ルビで事前学習 → 生成データで微調整」が良い。** 混ぜると文一致が約 3 pt 落ちた
（文学作品の地の文なので、短い文の何でもない語に文学的な読みを選んでしまう）。

---

## 6. train / dev の分割

```bash
uv run python scripts/split_train_dev.py \
  --train-out data/train_m5_split.jsonl --dev-out data/dev_r21.jsonl \
  --dev-exclude-chars '〇一二三四五六七八九十百千万零'
```

学習・評価スクリプトの既定値（`TrainConfig`・`compare_dev.py` など）と `infra/vast_sync.sh` は、
この 2 つのファイル名（`data/train_m5_split.jsonl` / `data/dev_r21.jsonl`）を使う（名前は開発時の経緯による）。
dev の監査結果（両方成立する読みの対応表）は `scripts/audit_dev.py` で `data/dev_audit_r21.jsonl` に作る。

- **語（`target_surface`）単位で分ける。** 文単位で分けると同じ語が両側に出て、実力を過大評価する
- 数詞を含む語は dev に回さない（`--dev-exclude-chars`。[R-21](pitfalls.md#r-21)）
- ★**データの構成を変えて比べるときは、`--dev-from` で dev を固定する。** 分割をやり直すと dev まで変わり、何が効いたか分からなくなる
- `--dev-out` に現行の dev を指定しない（上書きする）

1 文字の常用漢字（`方`・`後`・`山`）は、学習文の本文から排除できない（[R-20](pitfalls.md#r-20)）。

---

## 7. 学習

```bash
uv run python scripts/train.py --config configs/size-xs.yaml --seed 42 --out-dir checkpoints/size-xs-42
```

| 設定 | hidden × 層 | params | 備考 |
|---|---|---|---|
| `size-xs.yaml` | 128 × 3 | 2.07M | 最小。精度は大きい構成とほぼ同じ |
| `size-s.yaml` | 192 × 4 | 4.32M | |
| `size-m.yaml` | 256 × 4 | 6.97M | |
| `size-l.yaml` | 320 × 6 | 12.69M | |
| `tiny.yaml` | 256 × 4 | 6.97M | size-m と同じ構成で 10 epoch（開発初期の基本設定） |
| `tiny-cross.yaml` | 256 × 4 | 10.59M | CrossEncoder（論文準拠の構成） |
| `tiny-pretrained.yaml` | — | — | 事前学習エンコーダ（参照用。`hf` extra・CC BY-SA 4.0 のモデル） |
| `trans16.yaml` / `trans32.yaml` | 256 × 4 | — | size-m + 低ランクの遷移項（16 / 32 次元） |
| `smoke.yaml` | 256 × 4 | — | 過学習できるかを確かめる配線確認用 |

入力の既定は `data/train_m5_split.jsonl` / `data/dev_r21.jsonl`（`--train-path` / `--dev-path` で変えられる）。

- `latest.pt` を定期保存し、同じ `--out-dir` で起動すると optimizer と乱数の状態ごと再開する
- `best.pt` は dev で選ばれる
- ルビ事前学習 → 微調整は `--ruby-path data/ruby.jsonl` で事前学習し、`--init-from <事前学習の out-dir> --init-from-prefer latest` で微調整する
  （事前学習の `best.pt` は dev で選ばれており、事前学習の目的と基準が違う）
- ★**0.6 pt 未満の差は、seed を変えて（42 / 1337 / 2024）反復するまで報告しない**（[R-24](pitfalls.md#r-24)）

学習した checkpoint を dev で比べ、誤りを 3 分類する:

```bash
uv run python scripts/compare_dev.py checkpoints/a checkpoints/b
```

---

## 8. ONNX への書き出し

```bash
uv run python scripts/export_model.py --checkpoint checkpoints/size-xs-42 \
  --out models/size-xs-fp32 --quantize-out models/size-xs-int8
```

- 書き出したモデルは PyTorch の出力と突き合わせてから保存する
- INT8 は動的量子化。**読みエンコーダは FP32 のまま写す**（[R-26](pitfalls.md#r-26)）
- 推論は `lattice_g2p.runtime.OnnxG2P`（→ [README](../README.md#ニューラルモデルで読む)）

---

## 9. クラウド GPU（vast.ai）

学習規模（2〜13M params × 数千文）は小さく、RTX A4000 で 1 本 30 分・約 $0.05 だった（H100 は要らない）。
`infra/` に vast.ai 用のスクリプトがある（API キーは `.env` の `VAST_API_KEY`）。

```bash
infra/vast_search.sh                     # オファーを探す
infra/vast_launch.sh <OFFER_ID>          # インスタンスを作る（onstart で uv を入れる）
infra/vast_sync.sh push <INSTANCE_ID>    # コード・uv.lock・データを送る（インスタンス側でも uv sync で環境を再現）
infra/vast_sync.sh pull <INSTANCE_ID>    # checkpoint を回収する
infra/vast_teardown.sh <INSTANCE_ID>     # ★破棄する（起動している限り課金される）
```

- ★**CPU レイテンシはクラウドで測らない。** checkpoint を回収してローカルで ONNX 化し、ローカルで測る（[R-13](pitfalls.md#r-13)）
- 同じアカウントで他のプロジェクトのインスタンスが動いていることがある。**自分が起動したものだけを破棄する**（[R-14](pitfalls.md#r-14)）

---

## 10. データ形式と再現性

学習データは JSONL で 1 行 1 例。

```json
{
  "text": "今年の米は例年より出来がいいらしい。",
  "reading": "コトシノコメワレイネンヨリデキガイイラシイ",
  "target_start": 3, "target_end": 4,
  "target_surface": "米", "target_reading": "コメ",
  "source": "claude-generated",
  "gen_meta": {"prompt_version": "v1", "model": "...", "generated_at": "..."}
}
```

読みは発音形で書く（助詞の は は ワ）。LLM の出力は決定的でないので、少なくとも次を記録する。

- プロンプトの版（プロンプトは `prompts/` に置き、変えたら版を上げる）
- モデル ID と生成日時
- 各フィルタの通過率
