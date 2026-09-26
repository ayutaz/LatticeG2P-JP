---
name: auditing-evaluation
description: Use when accuracy looks stuck, when an error analysis is about to be trusted, or when building or changing an evaluation set in this project. Ten separate times the problem turned out to be the measurement rather than the model - this lists the specific failure shapes found, including two biases that quietly favored the comparison baseline, a comparison baseline whose output format was assembled wrongly (twice), an approximate reference that was off by 1.18 pt, and a home-grown metric that was 0.2-0.6 pt more lenient than the benchmark's official evaluator.
---

# 精度が頭打ちに見えたら、まず評価を疑う

★**このプロジェクトでは 10 回、評価側に原因があった**（M7 で 3 件、論文との照合で 1 件増えた）。
GPU を回す前に、誤り例を 15〜20 件**目視する**こと。

## 起きた 10 件

| # | 何が起きたか | 影響 |
|---|---|---|
| 1 | target span の読みを経路のノードから取り出していた | Accuracy 11% → 65% |
| 2 | 分割違いを不正解にしていた（損失 0 なのに 80.5%） | 100% 正解のモデルが 80.5% に見えた |
| 3 | dev のラベルが誤り / 両方成立（`法師→ボウシ`・`大麻→オオヌサ`） | 対象語 68.97% → **75.89%** |
| 4 | 四つ仮名を畳んでいなかった（`チヂム` vs `チジム`） | **+0.24 pt**。差の 5.1% |
| 5 | Hard Set の物差し自体（下記） | ★**2 件は比較相手に有利なバイアス** |
| 6 | 1 回の実行では 0.6 pt 未満を判定できない | ★**小さな報告がすべて無効に** |
| 7 | ★実 MeCab の読みを「助詞は pron」で組み立てて `を→オ` にした（M7） | 文一致を 42.82% と過小評価（正しくは 85.27%）。**R-22（OpenJTalk のアクセント記号）と同じ型の 2 回目** |
| 8 | ★INT8 の読みキャッシュが評価の履歴に依存した（M7, R-26） | 同じ 300 件で 94.33% / 94.67%。**評価の順序で数字が変わる** |
| 9 | ★比較の基準にした近似（U = MeCab 相当のつもり）が本物と 1.18 pt ずれていた（M7, R-27） | **近似を基準にするなら、本物も 1 度は測る** |
| 10 | ★自前の「対象語」の数え方が benchmark の公式評価ツールより 0.2〜0.6 pt 甘かった（R-28） | **公式の評価コードがあるなら、自前より先にそれを使う**。論文の数値は公式ツールで再現した |

## ★特に危険: 比較相手に有利なバイアス

**放置すると「勝った」と誤報告する。** 直すと OpenJTalk のほうが上がった。

### 5-a. 検証が比較相手の出力に依存していた

Hard Set の文を「OpenJTalk の読みと前後が一致するか」で検証していた。
`span_reading_from_morphemes` は span と重なる形態素を丸ごと含めるので、
**OpenJTalk が対象語を隣と融合させると必ず不一致になる。**

```
一二三  -> OpenJTalk は文脈によらず「123」と読む -> 構造的に必ず落ちる
心地よい -> 1 形態素になる -> 対象語の末尾が形態素の途中
```

→ **「OpenJTalk が大きく外す語ほど Hard Set から消える」**。
Hard Set の存在理由と正面から矛盾する。
**境界をまたぐ側だけチェックを諦める**形に直した。

### 5-b. 自由異読を片方に固定していた

`三階`(サンガイ/サンカイ)・`十回`(ジッカイ/ジュッカイ)・`七日`(ナノカ/ナヌカ)・
`日本`(ニホン/ニッポン) は**指す内容が同じで発音の差しかない**。
片方に固定すると**原理的に解けない問題**になる。

両方を正解にしたら、**得をしたのは OpenJTalk のほうが大きかった**
（数詞 +14.29 pt vs こちら +10.71 pt）。
**どちらが損をするかが偶然に左右されていた証拠**である。

## 評価セットを作る / 直すときのチェック

- [ ] **両方の読みが成立する文が無いか**（M3c で誤りの 33% がこれ）
- [ ] **自由異読を片方に固定していないか**（意味が同じで発音だけ違う組）
- [ ] **語そのものを話題にした文が無いか**（「〜という地名もある」は言及であって使用ではない）
- [ ] **検証が比較相手の出力に依存していないか** ← ★最も危険
- [ ] **正規化の不一致を吸収しているか**（長音・四つ仮名 → `phonetic_key`）
- [ ] **母数が小さい集合で「0 件」を根拠にしていないか**

★最後の項目: 四つ仮名は「dev 367 文で 0 件だから効かない」として
`phonetic_key` に入れなかったが、**benchmark 13,536 件では 33 件**あった。
**母数が 37 倍違えば「0 件」は「1% 未満」としか言っていない。**

## 誤り分析は 3 分類を伴わないと無意味

```
Oracle 失敗     辞書に正解がない  -> 辞書の問題。モデルを直しても改善しない
経路選択の誤り   正解は辞書にある  -> モデルの問題。ここが改善対象
正規化の不一致   バグ
```

```bash
uv run python scripts/run_eval.py --onnx models/<model>-int8   # 3 分類を出す
uv run python scripts/audit_dev.py                             # 疑わしいラベルを候補に挙げる
uv run python scripts/run_hard_set.py --onnx models/<model>-int8
```

★**監査は 1 段上のモデルで行う**（dev の 1/4 は haiku 生成なので、
haiku に監査させると同じ誤りを同じように見逃す）。

## ★指標が測れないものを直していないか

**Joyo benchmark の対象語は常用漢字の読みで、数詞を 1 件も含まない。**
数詞を Hard Set で +19.64 pt 直しても **benchmark では見えない。**

Hard Set が無ければ「効かなかった」と誤判断していた。
**直した弱点を、その指標が測れるか確認すること。**

## 対象語精度だけを見ない

```
             対象語     文一致
UnigramScorer 93.59%    24.02%   <- params 0。辞書コストのみ
こちら        94.3%     78.7%
OpenJTalk     96.85%    91.93%
```

★**params 0 のベースラインが対象語 93.59% を出す。**
**対象語精度だけを報告する評価は、辞書の力とモデルの力を分離できない。**
