# LatticeG2P-JP

辞書制約付きラティスと経路選択による、日本語のカナ読み推定（G2P）の研究実装。

A research implementation of dictionary-constrained grapheme-to-phoneme conversion for Japanese.
All segmentation × reading candidates from UniDic are kept in a lattice, and a scorer (dictionary costs,
UniDic's connection matrix, and/or a small neural network) picks the best path with a CRF trained only on
sentence-level readings. The research question was whether this beats OpenJTalk without a 100M-parameter
language model. **It does not:** the best configuration only ties OpenJTalk on target-word accuracy (and is worse
on whole-sentence readings), and that configuration is the dictionary with its connection matrix (equivalent to
MeCab + UniDic), not the neural scorer. Documentation is in Japanese.

## 目次

- [概要](#概要)
- [主な結果](#主な結果)
- [しくみ](#しくみ)
- [必要環境](#必要環境)
- [インストール](#インストール)
- [データの準備](#データの準備)
- [使い方](#使い方)
  - [辞書 + 連接行列で読む](#辞書--連接行列で読む)
  - [ニューラルモデルで読む](#ニューラルモデルで読む)
  - [評価する](#評価する)
  - [学習する](#学習する)
- [ディレクトリ構成](#ディレクトリ構成)
- [ドキュメント](#ドキュメント)
- [制約と既知の問題](#制約と既知の問題)
- [開発](#開発)
- [ライセンス](#ライセンス)
- [引用](#引用)

## 概要

日本語 TTS のフロントエンドでは、漢字の読み（`米を炊く → コメヲタク` / `米国政府 → ベイコクセイフ`）を決める必要がある。
このリポジトリは、辞書（UniDic）から「分割 × 読み」の候補をすべてラティスとして持ち、スコアラが最適な経路を選ぶ方式を実装・評価した。

- **形態素分割を正解データにしない。** 教師は「文」と「文全体の正しい読み」だけで、読みが一致する経路はすべて正解として扱う
- **読みを生成せず、辞書候補を順位付けするだけ。** 辞書にない読みは構造的に出力されない
- スコアラは差し替えられる: 辞書コストだけ / UniDic の連接行列 / 小型のニューラルネット（2〜13M params）とその組み合わせ

出発点は論文 *Dictionary-Constrained Grapheme-to-Phoneme for Unsegmented Languages from LLM-Annotated Data*
（arXiv 2609.19805）だが、再現を目的としていない。検証した問いは次の 2 つである。

1. 辞書制約があるなら、100M 級の言語モデルを使わずに OpenJTalk を大きく上回れるか → **上回れない。並ぶのが上限だった**
2. それは実用的な CPU レイテンシに収まるか → **収まる**（91 字 p50 2.85 ms）。ただし OpenJTalk より 1 桁遅い

**研究としては完結している。** コード・テスト・評価ツールを含む。学習済みモデル・学習データ・辞書は含まない。

## 主な結果

[Joyo-Kanji-Yomi Benchmark Parakeet Edition](https://huggingface.co/datasets/Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet)（13,536 文）。
benchmark の公式評価ツール（jkyb-eval）で測った値で、論文の表と同じ定義である。

| 構成 | Accuracy | Target PER | Sentence PER |
|---|---|---|---|
| OpenJTalk（pyopenjtalk） | 96.63 | 3.00 | 0.80 |
| MeCab + UniDic 3.1.1 | 96.72 | 2.75 | 1.16 |
| **このリポジトリ: 辞書 + 連接行列**（ニューラルなし） | **96.73** | 2.71 | 4.07 |
| このリポジトリ: ニューラル + 連接行列（6.97M・seed 3 本の平均） | 95.94 | 3.47 | 1.32 |
| このリポジトリ: ニューラルのみ（6.97M・seed 3 本の平均） | 93.70 | 5.52 | 2.20 |
| *参考: 論文の提案法（DeBERTa-base・LLM 注釈 234 万文）* | *99.62* | *0.32* | *0.14* |

- 自前で走らせた OpenJTalk は論文の値と 3 列とも一致した
- **品詞を保つ辞書 + UniDic の連接行列は MeCab + UniDic をほぼ完全に再現し、対象語の Accuracy で OpenJTalk と区別できない**。
  ただし文全体の読みには弱い（Sentence PER 4.07。助詞の は を ハ と読む）。文全体の読みは OpenJTalk が最も良い
- そこにニューラル経路選択（生成データ 6,554 文で学習）を足すと、対象語の Accuracy は下がる（seed 3 本とも）
- OpenJTalk との差の正体は、データ量・モデルサイズ・事前学習ではなく、**連接行列と品詞の情報**だった
- モデルサイズ 2〜13M で精度はほぼ変わらない。最小構成（2.07M・4.15 MB）は 91 字 p50 2.85 ms（Apple M1・1 スレッド）
- 人名はどの UniDic 系の構成も OpenJTalk を上回る（87〜94% vs 77%）

詳細は [docs/results.md](docs/results.md)。

## しくみ

```
入力文 ──> 辞書ラティス（考えうる「分割 × 読み」の候補をすべて持つ DAG）
              ↓
          スコアラ（辞書コスト / 連接行列 / ニューラル。候補ノードを一括で採点）
              ↓
          CRF / Viterbi（最尤経路を選ぶ）──> カナ読み
```

- **辞書**: UniDic CWJ 3.1.1 の発音形（`pron`）と仮名形（`kana`）の両方を候補にする。片方だけでは正解がラティスに入る割合（Oracle）が 32% 前後に落ちる。両方で 99.46%
- **学習**: CRF の分子（正解の読みを生成する経路の集合）を、(文字位置, カナ位置) の積グラフ上の forward で計算する。経路を列挙しない
- **連接行列**: UniDic の連接コストを固定の遷移項として足せる。辞書コストと組み合わせると MeCab と同じ目的関数になる
- **推論**: ニューラルスコアラは ONNX（INT8）で動き、読み表現はキャッシュする。Viterbi はホスト側（numpy）

詳細は [docs/architecture.md](docs/architecture.md)。

## 必要環境

- [uv](https://docs.astral.sh/uv/)
- Python **3.13.5**（`pyproject.toml` と `.python-version` で固定している。無ければ uv が入れる）
- 推論・評価は CPU だけで動く。学習には CUDA の GPU を推奨する
- ディスク: UniDic の zip が約 1.7 GB、使うファイルを展開して約 0.7 GB、辞書キャッシュが 61〜74 MB

開発と計測は macOS（Apple M1）、学習は Linux（クラウド GPU）で行った。

## インストール

```bash
uv sync --all-extras
uv run pytest
```

| extra | 中身 | 使う場面 |
|---|---|---|
| `dev` | pytest・ruff | テスト・lint |
| `onnx` | onnx・onnxruntime・onnxscript | ONNX への書き出しと推論 |
| `bench` | pyopenjtalk | OpenJTalk との比較 |
| `hf` | transformers | 事前学習エンコーダとの比較（参照用） |

★`uv sync --extra dev` だけだと onnxruntime と pyopenjtalk が入らず、それらのテストは**黙ってスキップされる**。

## データの準備

`data/` は gitignore されている。辞書と評価データは各自で取得する。

**1. UniDic CWJ 3.1.1**（GPL / LGPL / BSD 3-Clause のトリプルライセンス）

```bash
mkdir -p data/unidic-src
curl -L -o data/unidic-src/unidic-cwj-3.1.1-full.zip \
  https://clrd.ninjal.ac.jp/unidic_archive/cwj/3.1.1/unidic-cwj-3.1.1-full.zip
unzip -o data/unidic-src/unidic-cwj-3.1.1-full.zip -d data/unidic-src \
  "unidic-cwj-3.1.1-full/lex_3_1.csv" "unidic-cwj-3.1.1-full/matrix.bin" \
  "unidic-cwj-3.1.1-full/unk.def" "unidic-cwj-3.1.1-full/dicrc" "unidic-cwj-3.1.1-full/liceses/*"
```

MeCab + UniDic と比べるときは、同じ zip から `sys.dic`・`char.bin`・`unk.dic` も展開する。
`matrix.def`（テキスト版の連接行列・3.7 GB）は展開しなくてよい。

**2. Joyo-Kanji-Yomi Benchmark Parakeet Edition**（MIT）

```bash
mkdir -p data/benchmark
curl -L -o data/benchmark/common_kanji_source.jsonl \
  "https://huggingface.co/datasets/Parakeet-Inc/joyo-kanji-yomi-benchmark-parakeet/resolve/main/data/common_kanji_source.jsonl"
```

**3. 辞書キャッシュ**（初回だけ）

```bash
uv run python scripts/build_dict_cache.py                                          # 既定 → data/dic_cache.pkl
uv run python scripts/build_dict_cache.py --keep-pos --out data/dic_cache_pos.pkl  # 品詞を保つ（連接行列用）
```

★キャッシュは pickle なので、自分で作ったファイルだけを読み込むこと。

## 使い方

### 辞書 + 連接行列で読む

学習済みモデルは要らない。benchmark で最も精度が高かった構成（辞書 + 連接行列・品詞を保つ）。

```python
from pathlib import Path

from lattice_g2p.connection import ConnectionMatrix, unk_ids
from lattice_g2p.crf import viterbi_numpy
from lattice_g2p.decode import path_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.lattice import build_lattice
from lattice_g2p.scorer import UnigramScorer
from lattice_g2p.scorer.matrix import MatrixTransitionScorer

unidic = Path("data/unidic-src/unidic-cwj-3.1.1-full")
dic = Dictionary.load(Path("data/dic_cache_pos.pkl"))  # build_dict_cache.py --keep-pos
scorer = MatrixTransitionScorer(
    UnigramScorer(node_penalty=0.0),
    ConnectionMatrix.load(unidic / "matrix.bin"),
    weight=1.0,
    unk=unk_ids(unidic / "unk.def"),
)


def g2p(text: str) -> str:
    lattice = build_lattice(text, dic)
    scores = scorer.score(text, lattice.nodes).numpy()
    transitions = scorer.transitions(lattice.nodes).numpy()
    return path_reading(lattice, viterbi_numpy(lattice, scores, transitions))


print(g2p("米を炊く。"))        # コメヲタク
print(g2p("米国政府の発表"))    # ベイコクセイフノハッピョウ
print(g2p("生ビールを一杯"))    # ナマビールヲイッパイ
```

- 入力は正規化済みのテキストを想定している（数字・英字は範囲外。→ [制約と既知の問題](#制約と既知の問題)）
- 辞書と連接行列の読み込みに数秒かかる。行列（481 MB）は memmap で読む
- ★**この構成は語の読みの選択には強いが、文全体の読みには弱い。** UnigramScorer は発音形と仮名形を文脈で選べないので、
  助詞の は / へ を ハ / ヘ と読む（`今日は → キョウハ`）。文全体の読みが要るなら、ニューラル構成か OpenJTalk を使う

### ニューラルモデルで読む

[学習](#学習する)して ONNX に書き出したモデルを使う（学習済みモデルは同梱していない）。

```python
from pathlib import Path

from lattice_g2p.dictionary import Dictionary
from lattice_g2p.runtime import OnnxG2P

dic = Dictionary.load(Path("data/dic_cache.pkl"))
g2p = OnnxG2P(Path("models/size-xs-int8"), dic, num_threads=1)
print(g2p.g2p("今日は良い天気です。"))             # キョウワヨイテンキデス
reading, timing = g2p.g2p_with_timing("米国政府の発表")  # timing: 区間ごとの所要時間（ms）
```

- 出力はカタカナで、benchmark の表記規則に従う（助詞の は / へ は ワ / エ、を は ヲ。漢語の長音は ウ）
- `OnnxG2P` は読み表現をキャッシュするので、使ううちに速くなる

### 評価する

```bash
uv run python scripts/run_oracle.py        # Oracle: 辞書ラティスに正解読みが入る割合
uv run python scripts/run_eval.py          # UnigramScorer（ニューラルなし）と誤りの 3 分類
uv run python scripts/compare_systems.py --onnx models/size-xs-int8   # OpenJTalk と並べる（自前 + 公式の物差し）
uv run python scripts/run_benchmark.py --model models/size-xs-int8 --threads 1 --openjtalk   # CPU レイテンシ
```

`compare_systems.py` は既定で公式評価ツールの値も出す（初回はツールの取得にネットワークが要る。`--no-official` で止める）。
**論文や他の報告と比べるときは公式の値を使うこと。** 連接行列を足した構成と Hard Set の測り方は
[docs/evaluation.md](docs/evaluation.md#7-評価の実行)。

### 学習する

```bash
# 1. 学習データを作る（多音語の例文を LLM に書かせ、辞書ラティスでフィルタする）
uv run python scripts/select_polyphonic.py
uv run python scripts/make_batches.py --limit 200 --batch-size 20
#    各バッチについて prompts/subagent-generate-v1.md を渡して Claude Code のサブエージェントに書かせる
uv run python scripts/collect_batches.py
uv run python scripts/split_train_dev.py --train-out data/train_m5_split.jsonl --dev-out data/dev_r21.jsonl \
  --dev-exclude-chars '〇一二三四五六七八九十百千万零'

# 2. 学習する（GPU 推奨）
uv run python scripts/train.py --config configs/size-xs.yaml --seed 42 --out-dir checkpoints/size-xs

# 3. ONNX に書き出す（PyTorch の出力と突き合わせてから保存する）
uv run python scripts/export_model.py --checkpoint checkpoints/size-xs \
  --out models/size-xs-fp32 --quantize-out models/size-xs-int8
```

データ生成の API 経路・数詞の学習データ・青空文庫のルビ・クラウド GPU での学習は [docs/training.md](docs/training.md)。

## ディレクトリ構成

```
src/lattice_g2p/   本体（辞書・ラティス・CRF・スコアラ・学習・ONNX ランタイム・評価）
scripts/           CLI（辞書キャッシュ・評価・学習・書き出し・計測）
configs/           学習設定（size-xs〜l など）
prompts/           データ生成用のサブエージェントへの指示
infra/             クラウド GPU（vast.ai）の運用スクリプト
tests/             pytest
docs/              設計・評価・学習・結果・落とし穴
.claude/           Claude Code 用の skills と hooks
```

## ドキュメント

| | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 設計（辞書とラティス・スコアラ・CRF・ONNX・不変条件） |
| [docs/evaluation.md](docs/evaluation.md) | 評価指標・公式評価ツール・カナ正規化・評価データ・レイテンシの測り方 |
| [docs/training.md](docs/training.md) | 学習データの作り方・学習・書き出し |
| [docs/results.md](docs/results.md) | 結果の詳細・元論文との比較・続けるなら |
| [docs/pitfalls.md](docs/pitfalls.md) | 開発中に起きた問題と対処（R-01〜R-28。コード中のコメントはこの ID で参照している） |

## 制約と既知の問題

- **学習済みモデル・学習データ・Hard Set・辞書は含まない。** 学習データは LLM で生成したので、同じ手順でも同じものはできない
- **数字・英字・記号の読みは範囲外。** `2026年`・`RTX 5090` のような表記は前段のテキスト正規化で処理する前提
- **ニューラル構成の精度は OpenJTalk に届かない。** 実用には OpenJTalk か MeCab + UniDic を勧める
- 連接行列（481 MB）を使う構成はメモリを食う（memmap で読む）。品詞を保つ辞書ではラティスのノードも約 2 倍になる
- 文全体を見てから読むので、ストリーミングには向かない
- ラティスは分割を決め打ちしないが、正しい経路を選べるかはスコアラ次第である。例えば「外国人参政権」は、辞書 + 連接行列でも
  ニューラルでも `ガイコクニンジンセイケン`（外国 / 人参 / 政権）と読む

開発中に踏んだ問題（評価指標の定義ミス・比較相手の過小評価・INT8 の履歴依存など）は [docs/pitfalls.md](docs/pitfalls.md)。

## 開発

```bash
uv run pytest                          # テスト
uv run ruff check .                    # lint
python3 .claude/hooks/test_guard.py    # Claude Code の hook の判定テスト
```

CI（`.github/workflows/ci.yml`）は push / PR で `uv sync --locked --all-extras` の後に上の 3 つを実行する（CPU のみ・辞書と benchmark は不要）。

- Python の実行は `uv run` を通す（素の `python` / `pip` を使わない）
- 数値の差を報告するときは、学習を伴う比較なら seed を変えて反復する（同じ条件でも実行ごとに約 0.3 pt ずれる。[R-24](docs/pitfalls.md#r-24)）
- CPU レイテンシはローカルで測る（クラウドのインスタンスでは再現性が無い。[R-13](docs/pitfalls.md#r-13)）
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) で作業するときの指示は [CLAUDE.md](CLAUDE.md) と [.claude/](.claude/README.md) にある

## ライセンス

[Apache License 2.0](LICENSE)（Copyright 2026 ayutaz）。

依存するデータとモデルは、それぞれのライセンスに従う。どれもリポジトリには含めていない。

| | ライセンス | 備考 |
|---|---|---|
| UniDic CWJ 3.1.1 | GPL / LGPL / BSD 3-Clause から選択 | BSD 3-Clause を選べば寛容なライセンスのソフトウェアと組み合わせられる。辞書や、辞書から作ったキャッシュ・モデルを配布するときは、著作権表示・条件一覧・無保証条項を保持し、UniDic Consortium の名前を派生物の推奨・宣伝に使わないこと |
| Joyo-Kanji-Yomi Benchmark Parakeet Edition | MIT | 評価専用 |
| 青空文庫 | 著作権の消滅した作品だけを使う | 学習データ（ルビ）用。`fetch_aozora.py` が著作権フラグで絞る |
| `ku-nlp/deberta-v2-tiny-japanese-char-wwm` | CC BY-SA 4.0 | 精度の参照にだけ使った（`hf` extra）。配布する経路には使わない |

依存ライブラリ（PyTorch・ONNX Runtime・pyopenjtalk など）は、それぞれのライセンスに従う。

## 引用

このリポジトリの出発点になった論文:

```bibtex
@article{hu2026dictionary,
  title   = {Dictionary-Constrained Grapheme-to-Phoneme for Unsegmented Languages from LLM-Annotated Data},
  author  = {Hu, Rui and Zhan, Zhenpeng and Lin, Xiaolong},
  journal = {arXiv preprint arXiv:2609.19805},
  year    = {2026}
}
```

評価には Parakeet Inc. の Joyo-Kanji-Yomi Benchmark Parakeet Edition と公式評価ツール、辞書には国立国語研究所の UniDic を使った。
