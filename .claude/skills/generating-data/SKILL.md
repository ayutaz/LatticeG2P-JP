---
name: generating-data
description: Use when generating training or evaluation data with Claude Code subagents for this project - choosing which model to dispatch, writing the prompt, and deciding whether the mechanical filter can actually catch the errors that matter. Covers the measured yield differences between models, the "give the agent the verifier" finding, and the areas where filters silently pass broken data.
---

# サブエージェントでデータを作る

API キー（`ANTHROPIC_API_KEY`）は使わず、Claude Code のサブエージェントに生成させる（API 経路の `data/client.py` もあるが既定ではない）。

```bash
uv run python scripts/make_batches.py --limit 200 --batch-size 20
# 各 batch_NNN.jsonl についてサブエージェントを起動し batch_NNN.out.jsonl を書かせる
uv run python scripts/collect_batches.py    # 回収 -> フィルタ -> data/train.jsonl
```

## ★モデルを選ぶ前に、検証手段を渡せるか考える

Hard Set の生成で、`scripts/build_hard_set.py` の `check()` を import して
**自己検証するよう指示したら採用率が 50% → 100%** になった。

> **「安いモデルで足りるか」の前に「検証を回させたか」を確認する。**

```python
# エージェントに渡す検証コード
import importlib.util, json, collections
from pathlib import Path
spec = importlib.util.spec_from_file_location("bhs", "scripts/build_hard_set.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
reason, _ = m.check(row, allowed, dic, True)   # reason が None なら通過
```

## モデルの選び方

**品質に問題がなければ haiku → sonnet → opus の順。測ってから上げる。**

1. まず haiku で 1 バッチ試す
2. 歩留まり（`collect_batches.py` のラティス整合・最終歩留まり）を測る
3. ゲート（ラティス整合 > 95%、最終歩留まり > 85%）を割ったら 1 段上げる

### 実測された差

| タスク | haiku | sonnet |
|---|---|---|
| 学習データ生成（助詞規則の違反率） | 30.6% → **4.4%**（呼び出し側で規則を再掲） | — |
| Hard Set（機械検証の採用率） | **41.7〜66.7%** | **100%**（自己検証あり） |
| 数詞（歩留まり） | — | **100%** |

★**haiku は `text` を丸ごとカタカナで書く失敗もした**（214 件）。
`特牛駅ワ山陰線ノ小サナ駅デアル。` のような壊れ方で、
**UniDic はカタカナのエントリを持つのでラティス整合を通ってしまう。**

### 例外 1: 安いモデルが作ったものを監査するとき

`data/dev.jsonl`（M3c 当時の dev）の 1/4 は haiku 生成で、その**ラベルの妥当性を判定する**タスクに
haiku を使うと同じ誤りを同じように見逃す（循環する）。
**監査は 1 段上のモデルで行う**（dev の監査なら opus）。

### 例外 2: 評価データを作るとき

物差し自体が間違っていると測定の意味が無くなる。
**歩留まりより正しさを優先する**ので、最初から sonnet 以上でよい。

## ★エージェント起動時に最重要ルールを再掲する

プロンプトファイルに書いてあっても haiku は見落とす。
**起動プロンプトに助詞規則などを必ず再掲する。**

```
1. 助詞の「は」は ワ（「私は」→ ワタシワ）。「ハ」と書かない
2. 助詞の「へ」は エ／「を」は ヲ
3. 長音は表記どおり。「東京」は トウキョウ（トーキョー にしない）
4. 促音「ッ」・拗音「ャュョ」・撥音「ン」を省略しない
5. `target_start` は text.find() で機械計算する。目視で数えない
```

## ★フィルタが効かない領域がある

`本` には `ホン`・`ボン`・`ポン` がすべて辞書にあるので、
**`三本→サンホン` と書かれても機械的には通る。** 音便はフィルタで検出できない。

こういう領域では **知識（小さい表・高品質）と量（文・安価）を分離する**。
読みを事前の表で確定させ、エージェントには**文だけ書かせて判断の余地を無くす**。

→ `scripts/make_numeral_batches.py` / `prompts/subagent-numeral-v1.md`

### フィルタをすり抜けた実例

| 壊れ方 | なぜ通るか | 対策 |
|---|---|---|
| `三本→サンホン` | `ホン` も辞書にある | 読みを表で確定させる |
| `text` が丸ごとカタカナ | UniDic はカタカナのエントリを持つ | ひらがなが 1 文字も無い文を弾く |
| 語そのものを話題にした文 | 文としては正しい | 「読み」「音読」などを弾く |
| 自由異読を片方に固定 | 両方正しい | `free_variants` で両方を正解にする |

## 量を増やすときの注意

- ★**1 語 8 文では般化しない**（暗記に留まる）。**13 文以上**にして文脈を明示的に散らす
- ★**`collect_batches.py --held-out-chars` を全体に掛けない。** dev の該当例まで消える（419 → 361）
- ★**データ構成を変えて比較するときは `--dev-from` で dev を固定する**
