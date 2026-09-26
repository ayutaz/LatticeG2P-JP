# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクトの性質

**辞書制約付きラティス + 経路選択による日本語カナ読み推定（G2P）の研究実装。** 研究としては完結している。

- 問い: 辞書制約があるなら、100M 級の言語モデルを使わずに OpenJTalk を大きく上回れるか。それは実用的な CPU レイテンシに収まるか
- 答え: 上回れない。並ぶのが上限で、並んだのはニューラルを使わない「品詞を保つ辞書 + UniDic の連接行列」（= MeCab + UniDic）。
  ニューラル経路選択を足すと下がる。レイテンシは収まる（91 字 p50 2.85 ms）が OpenJTalk より 1 桁遅い
- 結果の詳細は `docs/results.md`。続きの候補は同 §10（**止める条件つき**。新しく始めるなら止める条件を先に決める）

出発点は論文 arXiv 2609.19805 だが、**論文値（99.62%）の再現を目標にしない**（公式コード・データが無く、届かなかった原因を判別できない）。
判断は「同じ評価コードで自前で走らせた OpenJTalk / MeCab との差」で行う。

ライセンスは Apache License 2.0（`LICENSE`）。

## コマンド

```bash
uv sync --all-extras                          # ★--extra dev だけだと onnxruntime / pyopenjtalk のテストが黙ってスキップされる
uv run pytest                                 # 全テスト
uv run pytest tests/test_crf_forward.py -k numerator   # 一部だけ
uv run ruff check .
python3 .claude/hooks/test_guard.py           # hook の判定テスト

uv run python scripts/build_dict_cache.py                                          # data/dic_cache.pkl
uv run python scripts/build_dict_cache.py --keep-pos --out data/dic_cache_pos.pkl  # 品詞を保つ（連接行列用）
uv run python scripts/run_oracle.py           # Oracle
uv run python scripts/run_eval.py             # UnigramScorer と誤りの 3 分類
uv run python scripts/compare_systems.py --onnx models/<name>   # OpenJTalk と比較（自前 + 公式の物差し）
uv run python scripts/run_m7.py --runs <run> ...                # 連接行列を足した構成（U / N）
uv run python scripts/run_benchmark.py --model models/<name> --threads 1   # CPU レイテンシ（★ローカルで）
```

`data/`・`checkpoints/`・`models/` は gitignore。辞書・benchmark の入手は README の「データの準備」。

## アーキテクチャ

```
入力テキスト（正規化済み）
  → Dictionary       UniDic（pron と kana の両方）を marisa-trie に
  → build_lattice    「分割 × 読み」候補の DAG
  → ContextEncoder   文 → 文字単位の文脈表現（1 文につき 1 回。ニューラル構成のみ）
  → NodeScorer       全ノードを一括で採点（Unigram / BiEncoder / CrossEncoder / MatrixTransitionScorer / OnnxScorer）
  → CRF / Viterbi    学習は (文字位置, カナ位置) の積グラフ上の CRF、推論は Viterbi（ホスト側）
  → カナ読み列
```

**形態素分割を正解データにしない。** 教師は文と文全体の読みだけで、読みが一致する経路はすべて正解。
スコアラは読みを生成せず辞書候補を順位付けするだけなので、辞書に無い読みは出力されない。
詳細は `docs/architecture.md`。コメント中の `R-xx` は `docs/pitfalls.md` の ID。

## 設計上の不変条件

詳細と根拠は `docs/architecture.md` §9（番号は共通）。変えるときはそちらとテストも直す。

1. 辞書の読みは UniDic の `pron` と `kana` の**両方**を採る（片方だけでは Oracle が 32% 前後）
2. カナ正規化は `src/lattice_g2p/kana.py` の**単一実装**を、辞書・データ生成・評価で共有する
3. `NodeScorer` は差し替え可能に保つ。単体ノードを受けるメソッドを作らない
4. ノードスコアリングは必ずバッチ化する（`for node in nodes: model(node)` は動くが 1 文で数百 ms）
5. CRF の分子は積グラフ上の制約付き forward で計算する（経路を列挙しない）
6. CRF / Viterbi は ONNX に含めず、ホスト側に置く
7. 辞書データをコミットしない。`.gitignore` の `/data/` は**先頭スラッシュ必須**（外すと `src/lattice_g2p/data/` が無視される）
8. 読みの由来（`source`: pron / kana / both / aug）を保つ
9. `phonetic_key` は比較専用（長音・四つ仮名を畳む）。出力する読みに使わない
10. 精度は文全体の読みで測る。対象語の span 単位は参考値
11. 空読みノードを 0 ベクトルにしない（`EMPTY_READING_ID`・`safe_mask`。全マスク行は attention が nan）
12. dev の語は、どのデータ源からも学習に入れない（生成データの分割とルビ側の `drop_held_out_spans` の両方）
13. 部分制約（ルビ）では span の境界をまたぐノードを採用しない
14. 遷移項を足しても積グラフは変えない。DP を辺で回し、`transitions=None` では従来の実装を通す
15. `Entry` にフィールドを足したら `_CACHE_FIELDS` と `from_entries` も直す（忘れると黙って既定値になる）
16. 対象語の読みを 1 本の経路から取り出さない
17. 仮名・漢字を含む表記に空読みを許さない
18. 連接行列を使うときは、連接 ID を持たないノード（補完・未知語）の扱い（`unk` か `unresolved_cost`）を明示し、品詞を保つ辞書を使う。0 は中立ではない

## 測るときの規則

- ★**0.6 pt 未満の差は、反復するまで報告しない。** 同じ条件でも別々の実行で約 0.30 pt ずれる（seed 3 本で SD 0.11 pt）。
  `scripts/train.py --seed 42`（1337 / 2024 でも回す）。学習しない比較（同じ checkpoint に行列を足す等）は決定的なので反復不要
- ★**論文や他の報告と比べるときは、benchmark の公式評価ツールの値を使う**（`compare_systems.py` / `run_m7.py` が既定で出す）。自前の「対象語」は 0.2〜0.6 pt 甘い
- ★**比較相手の出力も検証する。** OpenJTalk の `pron` はアクセント記号を含む（`baselines.clean_reading` / `find_unexpected_chars` を使う）。MeCab の読みは「助詞は pron、ただし を は kana、他は kana」。**「相手が弱い」という数字が出たときこそ自分の計測を疑う**
- ★**CPU レイテンシはローカルで測る。** クラウドのインスタンスでは再現性が無い（数値は「それらしく」出る）。**文長ごとに**出し、CPU・スレッド数を記録する
- 設計に書いた最適化（読みキャッシュなど）を実装したか、測る前に確認する
- ハイパーパラメータは dev（`data/dev_r21.jsonl`）で選び、benchmark では選ばない
- 評価は誤りの 3 分類（Oracle 失敗 / 経路選択 / 正規化の不一致）を伴わせる。精度が頭打ちに見えたら評価側（dev のラベル・物差し）を先に疑う
- INT8 は読みエンコーダを FP32 のまま写す（`export_onnx.KEEP_FP32`。動的量子化 + キャッシュで出力が履歴に依存する）

## データの規則

- 現行の分割: `data/train_m5_split.jsonl` / `data/dev_r21.jsonl`（`TrainConfig` と各スクリプトの既定値）。監査の対応表は `data/dev_audit_r21.jsonl`
- ★データ構成を変えて比べるときは `split_train_dev.py --dev-from data/dev_r21.jsonl` で dev を固定する。`--dev-out` に現行の dev を指定しない
- ★`collect_batches.py --held-out-chars` を全体に掛けない（dev の該当例まで消える）。新しいバッチだけに掛ける
- held-out にしてよいのは推論できる語だけ（数詞の促音化は語彙的知識なので dev に入れない）
- 青空文庫のルビは学習専用（文全体の読みが無いので評価に使えない）。混ぜるより「ルビで事前学習 → 生成データで微調整（`--init-from ... --init-from-prefer latest`）」
- ラティス整合フィルタが効かない領域がある（`三本 → サンホン` も通る）。そこでは読みを表で確定させ、エージェントには文だけ書かせる

## 実行環境の規則

- **Python は uv で管理する。** 素の `python` / `pip` を使わない（`uv run python ...`・`uv add ...`）。クラウド側でも `uv sync` で再現する。Python は 3.13.5 に固定
- **GPU が要る処理はクラウド（vast.ai）で行う。** ローカルで GPU 学習をしない。API キーは `.env` の `VAST_API_KEY`（gitignore 済み。コミット・表示しない）。
  起動したインスタンスの ID は `/tmp/my_instance` に記録し、**自分が起動したものだけを破棄する**
- **学習データの生成は Claude Code のサブエージェントで行う**（API 経路の `data/client.py` もあるが既定ではない）
  - モデルは安い順（haiku → sonnet → opus）。1 バッチ試して歩留まりを測り、ゲート（ラティス整合 > 95%・最終 > 85%）を割ったら上げる
  - 起動時の指示で最重要ルール（助詞の読みなど）を再掲する。検証手段（`check()` など）を渡せるなら渡す
  - 安いモデルが作ったラベルの監査は 1 段上のモデルで行う（同じモデルは同じ誤りを見逃す）
  - 件数はエージェントの報告ではなくファイルから数える

## skills と hooks

`.claude/` にこのリポジトリ固有の skills と hooks がある（→ `.claude/README.md`）。

| skill | いつ読まれるか |
|---|---|
| `measuring-differences` | 数値の差を報告・比較・判断するとき |
| `vast-training` | vast.ai で学習を回すとき |
| `generating-data` | サブエージェントでデータを作るとき |
| `auditing-evaluation` | 精度が頭打ちに見えたとき / 評価セットを作るとき |

hooks（`PreToolUse` / Bash）は 3 つを deny する（確認は求めない。理由を読んでコマンドを直して続ける）:
素の python / pip・ssh 越しのレイテンシ計測・`/tmp/my_instance` に無い ID の `vastai destroy`。

## ドキュメント

| | |
|---|---|
| `README.md` | 概要・結果・インストール・使い方 |
| `docs/architecture.md` | 設計と不変条件 |
| `docs/evaluation.md` | 指標・公式評価ツール・正規化・評価データ・レイテンシの測り方 |
| `docs/training.md` | 学習データ・学習・書き出し・vast.ai |
| `docs/results.md` | 結果・元論文との比較・続きの候補・開発の経緯（M0〜M7） |
| `docs/pitfalls.md` | 開発中に起きた問題（R-01〜R-28） |

## 用語

| 用語 | 意味 |
|---|---|
| ラティス | 「分割 × 読み」候補を辺とする DAG。頂点は文字位置 |
| ノード | ラティスの辺 `(start, end, 表記, 読み)`。表記が同じでも読みが違えば別ノード |
| Oracle | 辞書ラティスに正解読みが存在するか（スコアラを使わない） |
| 正解経路集合 G | 連結した読みが正解カナ列と一致する経路の集合 |
| 対象語 | benchmark の各文にある評価対象の語 |
| U / U_pos / N0 / N / N_pos | 辞書 + 行列（品詞を潰す / 保つ）・ニューラル・ニューラル + 行列（品詞を潰す / 保つ）。`docs/results.md` §1 |
