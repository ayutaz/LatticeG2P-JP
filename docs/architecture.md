# 設計

辞書制約付きラティス + 経路選択による日本語カナ読み推定の設計。
実装は `src/lattice_g2p/`。落とし穴の ID（R-xx）は [pitfalls.md](pitfalls.md) を参照。

**中核の考え方は、形態素分割を正解データにしないこと。** 教師は「文」と「文全体の正しい読み」だけで、
正しい読みを生成する経路をすべて正解として扱う。スコアラは読みを**生成**せず、辞書候補の**ランキング**だけを行うので、
辞書にない読みを構造的に出力できない。

## 目次

1. [全体像](#1-全体像)
2. [データ構造](#2-データ構造)
3. [辞書とラティス](#3-辞書とラティス)
4. [ノードスコアラ](#4-ノードスコアラ)
5. [CRF と Viterbi](#5-crf-と-viterbi)
6. [数値安定性](#6-数値安定性)
7. [ONNX とランタイム](#7-onnx-とランタイム)
8. [ディレクトリ構成](#8-ディレクトリ構成)
9. [設計上の不変条件](#9-設計上の不変条件)

---

## 1. 全体像

```
入力テキスト（正規化済み）
  → Dictionary       UniDic を marisa-trie にロード
  → build_lattice    「分割 × 読み」候補の DAG を構築
  → ContextEncoder   文全体 → 文字単位の文脈表現（1 文につき 1 回）
  → NodeScorer       各ノード (span, reading) にスコア（★全ノード一括）
  → CRF / Viterbi    学習は CRF、推論は Viterbi で最尤経路を選ぶ
  → カナ読み列
```

`ContextEncoder` は 1 文につき 1 回しか走らない。ノード数 N に比例して走るのは `NodeScorer` なので、
ここを**必ずバッチ化**する（→ §7-1）。ニューラルを使わない構成（`UnigramScorer` + 連接行列）では
`ContextEncoder` は無く、辞書コストと連接コストだけで経路を選ぶ（→ §4-6・§4-8）。

---

## 2. データ構造

### 2-1. Node

ラティスの辺。**表記が同じでも読みが違えば別ノード**になる。

```python
@dataclass(frozen=True)
class Node:
    start: int         # 入力文字列上の開始位置（文字単位）
    end: int           # 終了位置（exclusive）
    surface: str       # 表記     例: "米"
    reading: str       # 読み     例: "コメ"（正規化済みカタカナ）
    entry_id: int      # 辞書エントリ ID。-1 は未知語フォールバック
    pos: str           # 品詞
    cost: int = 0      # UniDic の生起コスト（小さいほど出やすい）
    source: str = ""   # 読みの由来: "pron" / "kana" / "both" / "aug"
    left_id: int = 0   # 連接 ID（未知語・補完ノードは 0）
    right_id: int = 0
```

- `cost` は辞書事前分布と `UnigramScorer` に使う
- `source` は、pron 由来と kana 由来の読みが**同じコストになって区別できない**ので、スコアラの特徴にする（→ §3-1）
- 読みが空文字のノードは句読点などの記号を表す。CRF では文字位置だけを進め、カナ位置を進めない

### 2-2. Lattice

文字位置 `0..L` を頂点とし、`Node` を辺とする DAG。`nodes_starting_at[i]` / `nodes_ending_at[i]` で引ける。

- 位置 0 から L への経路が少なくとも 1 本ある（未知語フォールバックで保証。→ §3-6）
- すべてのノードで `0 <= start < end <= L`

### 2-3. TrainingExample / PreparedExample

学習データは 2 段階で持つ。

| | 中身 | 場所 |
|---|---|---|
| `TrainingExample` | 文・文全体の読み・対象語の span と読み・生成メタデータ | `data/schema.py` |
| `PreparedExample` | ラティスのノードと、CRF の分母・分子の遷移構造を前計算したもの（→ §5-4） | `data/prepare.py` |

ルビ由来の例は文の一部の読みしか持たないので、`PreparedExample.gold_reading` は `None` になる（→ §5-6）。

---

## 3. 辞書とラティス

### 3-1. `pron` と `kana` の両方を読み候補にする

UniDic の `pron`（発音形）と `kana`（仮名形）は一致せず、**どちらが benchmark の表記と合うかは語によって違う。**

| 表記 | `pron` | `kana` | benchmark | 正しいのは |
|---|---|---|---|---|
| は（係助詞） | **ワ** | ハ | ワ | `pron` |
| へ（格助詞） | **エ** | ヘ | エ | `pron` |
| を（格助詞） | オ | **ヲ** | ヲ | `kana` |
| 米 | ベー | **ベイ** | ベイ | `kana` |
| 注文 | チュー… | **チュウ**… | チュウモン | `kana` |

| 辞書 | Oracle（benchmark 全体） |
|---|---|
| `pron` のみ | 31.51% |
| `kana` のみ | 33.62% |
| **`pron` + `kana`** | **99.46%** |

**辞書は候補を漏らさないことを優先し、選択はスコアラに任せる。**
両方入れると同じ span に同じコストのノードが 2 つ並ぶので、由来（`source`）を特徴として持たせる。

### 3-2. 空読みの規則

UniDic は顔文字の部品として `人` のような漢字を**読みなし**で登録している。そのまま使うと
「人」を読み飛ばす経路ができ、文字を脱落させた読みを出力できてしまう。
**仮名・漢字を含む表記には空読みを許さない**（`allows_empty_reading()`。品詞ではなく文字種で判定する）。

```
。、「」…  → 空読みを許す（読みに寄与しない記号）
人 を 米   → 空読みを許さない
```

### 3-3. 辞書の構築

```
UniDic CWJ 3.1.1  lex_3_1.csv（233 MB）
  ├─ 表記 / pron / kana / 品詞 / コスト / 連接 ID を抽出（874,018 entry）
  ├─ カナ正規化（kana.py。評価側とまったく同じ関数）
  ├─ 重複排除: 既定は (表記, 読み)。コストは最小値、source は統合
  ├─ 仮名・漢字を含む表記の空読みを除去
  └─ 複合語から単漢字の読みを補完（dict_augment.py。+11,185 件。→ §3-5）
  ↓
885,203 entry → marisa-trie（共通接頭辞検索）
  ↓
data/dic_cache.pkl（61 MB。CSV のパースに数秒かかるのでキャッシュする）
```

CSV の形式は `surface, left_id, right_id, cost, f[0]..f[28]` で、`f[9]` が `pron`、`f[20]` が `kana`
（列番号は `dicrc` の feature 定義から決まる）。

★**辞書側と評価側のカナ正規化は必ず同じ関数を通す。** 別々に実装すると、Oracle が落ちた原因が
正規化なのか辞書なのか判別できなくなる（[R-10](pitfalls.md#r-10)）。

★キャッシュは pickle なので、**自分で作ったファイルだけを読み込むこと。**

### 3-4. 品詞を保つ辞書（`keep_pos`）

既定の重複排除は (表記, 読み) で 1 つに潰し、**最小コストのエントリの品詞と連接 ID だけを残す**。
ニューラルスコアラは品詞を使わないので問題にならないが、連接行列（→ §4-8）は品詞で引くので情報が壊れる
（[R-27](pitfalls.md#r-27)。「実施さ(れる)」の `さ` が終助詞だけになる）。

```bash
uv run python scripts/build_dict_cache.py --keep-pos --out data/dic_cache_pos.pkl
```

`keep_pos=True` は重複排除のキーを (表記, 読み, left_id, right_id) にする。
エントリは 1,072,659 件（74 MB）、ラティスのノードは約 2 倍（平均 83 → 164）になる。Oracle は変わらない。
**連接行列を使うなら品詞を保つ辞書を使う。** 既定の辞書とラティスは `keep_pos` を足す前と 1 ビットも変わらない。

### 3-5. 単漢字読みの補完

論文は mpaligner で「単独のエントリとして登録されていない漢字の読み」を約 16,600 件抽出している。
ここでは**既知の読みを使った制約伝播**で決定的に抽出する。

```
撮了 → サツリョウ で「了 → リョウ」が既知なら、「撮 → サツ」が一意に決まる
```

全漢字の複合語のうち、既知の読みで説明できない文字がちょうど 1 つあり、その割り当てが一意に決まるものだけを採る。
**推測で辞書を汚さない。**

| 構成 | entries | Oracle（全体） | Oracle（スコープ内） | 平均ノード数 |
|---|---|---|---|---|
| 補完なし | 874,018 | 99.14% | 99.58% | 52.4 |
| **1 回**（採用） | **885,203** | **99.46%** | **99.90%** | **83.3** |
| 2 回 | 887,587 | 99.47% | 99.91% | 90.9 |

2 回目は Oracle が +0.01 pt しか改善しないのにノードが 9% 増えるので、1 回にした。

### 3-6. ラティスの構築

各位置から共通接頭辞検索で辞書エントリを引き、読みごとにノードを作る（多音語は複数ノードになる）。

**未知語フォールバック:** 辞書に一致しない文字があると経路が途切れるので、すべての位置に 1 文字ノードを用意する。

| 文字種 | フォールバックの読み |
|---|---|
| ひらがな・カタカナ | そのままカタカナ化 |
| 漢字 | 辞書から得られる単漢字読みの全候補。無ければ `UNK` マーカ（`〇`） |
| 記号（仮名・漢字を含まない） | 空読み |
| 数字・英字 | 前段のテキスト正規化で処理済みの前提（[R-05](pitfalls.md#r-05)） |

`UNK` マーカを空読みにしてはいけない（その文字を読み飛ばす経路が正解になりうる）。
フォールバックノードは `entry_id = -1`・`cost = 10000` で、辞書エントリより明確に不利にしてある。

ラティスの大きさは問題にならない。benchmark 全体で平均 83.3 ノード・最大 236 ノード、62 字で構築 0.3 ms 程度。

---

## 4. ノードスコアラ

### 4-1. インターフェース

スコアラは必ずこのインターフェースの背後に置く。**精度・レイテンシ・サイズを比べることが目的の一つなので、ここを固定しない。**

```python
class NodeScorer(Protocol):
    def score(self, text: str, nodes: list[Node]) -> Tensor:
        """全ノードのスコアを一括で返す。 shape: (len(nodes),)"""
```

遷移項を持つスコアラは、加えて `transitions(nodes) -> (N, N)` を持つ（→ §5-9）。
評価スクリプトは `getattr(scorer, "transitions", None)` で扱うので、スコアラを足してもスクリプトは変わらない。

| 実装 | params | 中身 |
|---|---|---|
| `UnigramScorer` | 0 | 辞書コストだけ（→ §4-6） |
| `BiEncoderScorer` | 2.07〜12.69M | 軽量版。span 表現と読み表現の内積（→ §4-3） |
| `CrossEncoderScorer` | 10.59M | 論文準拠の構成（→ §4-2） |
| `MatrixTransitionScorer` | 0 | 任意のスコアラに UniDic の連接行列を足す（→ §4-8） |
| `OnnxScorer` | — | ONNX に書き出した BiEncoder（→ §7） |

### 4-2. CrossEncoderScorer（論文準拠）

```
文全体 → ContextEncoder → H (L, d)            ※1 文につき 1 回
span[start:end] の H ─┐
                      ├→ Transformer Encoder ×2 → Decoder ×2（H 全体と cross attention）→ pooling → s(v)
reading → ReadingEncoder ┘
```

論文と同じ構成。軽量版と同条件で比べるための基準として置いた。

### 4-3. BiEncoderScorer（軽量版）

```
span 表現:     pool(H[start:end]) → Linear → u ∈ R^d
reading 表現:  ReadingEncoder(reading) → v ∈ R^d     ← ★入力文に依存しない
score:         s(v) = <u, v> / sqrt(d) + 読みの由来などの特徴 + 辞書事前分布
```

**読み表現は入力文に依存しないので、読みごとに 1 度だけ計算してキャッシュできる。**
推論時はノードごとに内積 1 回で済む。これが BiEncoder を選んだ理由で、キャッシュが無いと 91 字の文で 9.1 倍遅い
（57 ms → 6.29 ms。[R-06](pitfalls.md#r-06)）。

span の平均プーリングは累積和の差で取る。`(N, L)` のマスクを掛けて足すと、391 × 91 × 256 の中間テンソル（36 MB）が毎回できる。

### 4-4. ReadingEncoder（埋め込みテーブルにしない）

読みの表現を (表記, 読み) ごとの埋め込みテーブルで持ってはいけない。

```
151,000 entry × 256 次元 × 4 byte ≒ 155 MB   ← 軽量化の目標を単独で壊す
```

**カナ文字列を小さな文字単位エンコーダに通す**（カナ約 100 種の文字埋め込み → 小型 Transformer → pooling）。
パラメータは文字種数に比例して小さく、結果は読みごとにキャッシュでき、未知の読みにも般化する。
空読み（句読点）には専用 ID（`EMPTY_READING_ID`）を割り当てる（[R-18](pitfalls.md#r-18)）。

### 4-5. ContextEncoder

文字位置とラティスの位置を 1 対 1 に対応させるため、**文字単位**のエンコーダを使う（サブワード単位のモデルは使わない）。

出荷経路は**スクラッチの文字単位 Transformer**（`ScratchCharEncoder`）。文字単位の事前学習モデル
（`ku-nlp/deberta-v2-tiny-japanese-char-wwm`）は CC BY-SA 4.0 なので、精度の上限を測る参照にだけ使った
（`PretrainedCharEncoder`・`configs/tiny-pretrained.yaml`。[R-08](pitfalls.md#r-08)）。
2.07〜12.69M で精度はほぼ動かない（→ [results.md](results.md#5-精度レイテンシサイズのカーブ)）。

### 4-6. UnigramScorer（ニューラルなし）

辞書コストだけでスコアを作る。CRF・Viterbi・評価が正しく繋がっているかを確かめる基準でもある。

```
score(v) = −cost / cost_scale − node_penalty + source_bonus[source]   （未知語は −unk_penalty）
```

- `length_bonus` は**原理的に無意味**。経路上のノード長の総和は必ず文字数 L なので、経路によらない定数になる。長い語を優先したいなら `node_penalty`
- pron 形と kana 形は同じコストなので、`source_bonus` で表記の流儀を選ぶ（既定は kana 形をわずかに優先。助詞の は が ハ になる）
- 連接行列と組み合わせるときは `node_penalty = 0` にする（連接コストが同じ役割を果たす）

### 4-7. 辞書事前分布

ニューラルスコアラは、辞書コストを重み付きで足す（`dict_prior_weight`。初期値 0.05 で学習する）。

```
s(v) = ニューラルのスコア + dict_prior_weight × (−cost / 100)
```

UniDic のコストは「その表記のその読みがどれくらい一般的か」という強い事前分布で、これが無いと `虎 → ドラ` のように
低頻度の読みを根拠なく選ぶ。dev の対象語で +4.25 pt。事前学習エンコーダの利得の大半もこれと重複していた。

### 4-8. 連接行列（MatrixTransitionScorer）

UniDic の連接行列（`matrix.bin`・15,626 × 15,388・固定）を、任意のスコアラに**固定の遷移項**として足す（`scorer/matrix.py`）。

```
score(v)          = base(v) + weight × (−(BOS → v と v → EOS の連接コスト) / cost_scale)
transitions[u, v] = base の遷移（あれば） + weight × (−cost(u → v) / cost_scale)
```

- `UnigramScorer(node_penalty=0)` に `weight = 1.0` で足すと、Viterbi は **MeCab の目的関数（Σ 単語コスト + Σ 連接コスト）を最小にする経路**を返す（`tests/test_matrix_scorer.py` が総当たりと突き合わせている）
- BOS / EOS は unary に畳み込むので、CRF の DP は変えない
- 行列は `ConnectionMatrix.load()` で memmap する。`cost(prev → next) = M[prev.right_id + lsize × next.left_id]`。軸の向きは `scripts/verify_matrix.py` で `matrix.def`（テキスト）と突き合わせられる
- ★**補完ノード（`aug`）と未知語ノードは連接 ID を持たない**（0 は BOS / EOS と衝突する）。`unk`（`unk.def` の文字種の ID）か `unresolved_cost` を与えないと止まる。**0 は中立ではない**（連接コストの平均は 5,319。0 にすると補完ノードを大きく優遇する）
- ★**品詞を保つ辞書（§3-4）と組み合わせる**

### 4-9. 低ランクの遷移項（TransitionScorer）

連接 ID ごとの埋め込みの内積 `<right_emb[u], left_emb[v]>` で遷移を学習する（16 / 32 次元）。
モデルサイズはほぼ増えない（+0.5 MB）が、benchmark では効果を検出できなかった（+0.07 pt / SE 0.28）。
連接行列そのもの（§4-8）は効いたので、フルランクの行列を低ランクで近似できなかったと考えられる。

---

## 5. CRF と Viterbi

### 5-1. 損失

$$
L_{CRF} = \log \sum_{\pi \in P} e^{s(\pi)} - \log \sum_{\pi \in G} e^{s(\pi)}
\qquad s(\pi) = \sum_{v \in \pi} s(v)
$$

P は全経路、G は**連結した読みが正解カナ列と一致する経路**の集合。
「外国人 / 参政権」と「外国 / 人 / 参政 / 権」のどちらが正しい分割かは問わず、読みが一致すればすべて正解として扱う。

$$
L = L_{CRF} + \lambda L_{margin}
$$

### 5-2. 分母: DAG の forward

```python
alpha[0] = 0.0
for j in range(1, L + 1):
    alpha[j] = logsumexp([alpha[v.start] + score[v] for v in nodes_ending_at[j]])
Z = alpha[L]
```

### 5-3. 分子: 積グラフ上の制約付き forward

G を列挙すると経路数が指数的に爆発する。**(文字位置 i, カナ位置 k) の積グラフ上で forward を回す。**

状態 `(i, k)` は「入力の先頭 i 文字を消費し、正解カナ列の先頭 k 文字を生成した」を表す。
ノード `v = (start=i, end=j, reading=r)` は、`Y[k : k+len(r)] == r` のときだけ `(i, k) → (j, k + len(r))` の辺になる。

```python
beta[(0, 0)] = 0.0
for i in range(L):
    for k in reachable_kana_pos[i]:
        for v in nodes_starting_at[i]:
            if Y.startswith(v.reading, k):
                beta[(v.end, k + len(v.reading))] = logaddexp(..., beta[(i, k)] + score[v])
numerator = beta[(L, len(Y))]
```

最悪 O(|nodes| × |Y|) だが、k は i でほぼ決まるので、到達可能な状態は非常に少ない。

### 5-4. 遷移構造は前処理でキャッシュする

スコアはパラメータに依存するので forward は毎ステップ要るが、**どのノードがどの辺になるか**は文と正解読みだけで決まる。
前処理で一度計算して `PreparedExample`（`denom_edges` / `numer_edges`）に持たせる。学習ループは segment logsumexp だけになる。

### 5-5. 正解経路上で対象語が取りうる読み

対象語の読みを**任意の 1 本の経路から取り出してはいけない。**

```
「心を固める」 target=「固」
唯一の正解経路が「固める → カタメル」の 1 ノード → span の境界が無く、「固」の読みを分離できない
```

- `target_readings_on_gold_paths()`: 文字位置がちょうど対象語の境界にある状態の組から、カナ位置の区間を求める。境界を持つ経路が無ければ空集合
- `covering_kana_windows()`: 分離できないときも「このカナ範囲のどこかに対象語の読みがある」という緩い制約を課す

データ生成のフィルタはこの 2 段階で判定する（1 段目だけでは歩留まりが 24 pt 落ちる）。

### 5-6. 部分制約（文の一部だけ読みが分かる場合）

青空文庫のルビは文の一部にしか付かない。

```
彼は｜生真面目《きまじめ》な男だった   → spans = [(2, 6, "キマジメ")]  それ以外の読みは未知
```

状態を `(文字位置 i, その位置を含む span 内で消費したカナ数 k)` に一般化する。span の外では k が 0 に固定され
（= ラティスそのもの）、span の中は §5-3 と同じになる。**分子と分母の差が span の中に閉じ込められる。**

```python
build_constrained_graph(lattice, gold) == build_partially_constrained_graph(lattice, [(0, L, gold)])
```

★**span の境界をまたぐノードは採用しない。** 「米国 → ベイコク」の 1 ノードは、「米」だけのルビ（ベイ）に対して
カナのどこまでが「米」の読みか決められない。推測で緩めると誤った経路が正解集合に入る。
落とした経路も分母には残るので、出力する読みは変わらない。1 つのルビが使えなくても、同じ文の他のルビは使える。

### 5-7. Oracle = 「G が空でないか」

G が空なら、そのラティスでは正解読みを一切生成できない。これがそのまま **Oracle 評価**であり、
**学習データのフィルタ（ラティス整合チェック）と同じ処理**である。G が空の例は損失が −∞ になるので必ず除く。

### 5-8. Margin loss

同じ span 上の正解ノード v_g と不正解ノード v_c の組について `max(0, m − (s(v_g) − s(v_c)))`。
v_g は分子の forward で到達可能だったノードとして得られる。

### 5-9. Viterbi と遷移項

推論は logsumexp を max に置き換えた Viterbi（`crf.viterbi` / numpy 版 `crf.viterbi_numpy`）。

**遷移項（pairwise）があるときは、状態を増やさずに辺で DP を回す**（`crf._edge_pass`）。
辺は必ず文字位置を進めるので、辺の始点の昇順がそのままトポロジカル順になる。積グラフ（`ConstrainedGraph`）は変えない。
`transitions=None` のときは遷移項が無い実装をそのまま通るので、既存の数値は 1 ビットも変わらない。

---

## 6. 数値安定性

- `logsumexp` は max を引く実装を使う（`torch.logsumexp`）
- 到達不能な状態は `-inf` ではなく大きな負の有限値（`-1e9`）で持つ。`-inf - (-inf) = nan` を避ける
- 全位置がマスクされた attention 行は `nan` になる。`safe_mask` で避ける（[R-18](pitfalls.md#r-18)）
- 学習初期は分母と分子が近く損失が 0 付近になる。「損失が下がらない」と「実装のバグ」を区別するため、`UnigramScorer` で CRF が正しく動くことを先に確かめる

---

## 7. ONNX とランタイム

### 7-1. ノードスコアリングは一括で

```python
# ✗ 動くが 1 文で数百 ms〜1 秒を超える
for node in lattice.nodes:
    score = model(node)

# ✓
scores = model.score(text, lattice.nodes)   # 1 回の呼び出しで (N,)
```

### 7-2. グラフの分割と、ホスト側の Viterbi

可変長のノード数を扱うため、グラフを分けて書き出す（`export_onnx.py`）。

| グラフ | 入力 | 出力 |
|---|---|---|
| `context_encoder.onnx` | 文字 ID 列 | 文脈表現 H (L, d) |
| `reading_encoder.onnx` | 読みの ID 列 | 読み表現（★ランタイムがキャッシュする） |
| `node_scorer.onnx` | H・span 境界・読み表現・特徴 | スコア (N,) |
| `transition.onnx` | 連接 ID | 遷移スコア (N, N)（低ランクの遷移項を持つモデルだけ） |

**CRF / Viterbi は ONNX に含めず、ホスト側（numpy）に置く。** DAG の構造が動的だからで、他言語へ移植するときもこの分担を保つ。
書き出したモデルは PyTorch の出力と突き合わせてから保存する（`scripts/export_model.py`）。

ランタイムは `runtime.OnnxG2P`（`g2p(text)`・区間ごとの所要時間を返す `g2p_with_timing(text)`）と、
評価スクリプト用の `runtime.OnnxScorer`。

### 7-3. INT8 量子化

`export_onnx.quantize()` は動的量子化で INT8 に変換する。ただし**読みエンコーダだけは FP32 のまま写す**（`KEEP_FP32`）。
動的量子化は尺度をバッチの min / max で決めるので、読みキャッシュと組み合わせると出力が評価の履歴に依存する（[R-26](pitfalls.md#r-26)）。

### 7-4. レイテンシの内訳

レイテンシは合計だけでなく内訳で取る（`g2p_with_timing`・`scripts/run_benchmark.py`）。

```
lattice   ラティス構築
context   文脈エンコーダ          ← 最小構成でも半分を占める（91 字で 53%）
scoring   ノードスコアリング
viterbi   Viterbi
```

ノードスコアリングが支配的なら、バッチ化の失敗を疑う。

---

## 8. ディレクトリ構成

```
src/lattice_g2p/
├── kana.py               ★カナ正規化の単一実装（normalize_kana）と比較専用の phonetic_key
├── dictionary.py         UniDic ローダ・marisa-trie・キャッシュ（keep_pos）
├── dict_augment.py       複合語からの単漢字読みの補完
├── lattice.py            Node / Lattice / build_lattice
├── crf.py                積グラフ・forward・Viterbi（遷移項つきは辺の DP）
├── connection.py         UniDic の連接行列（matrix.bin）と unk.def
├── decode.py             経路 → 読み列
├── scorer/               NodeScorer の実装（unigram / bi_encoder / cross_encoder / matrix / transition）
├── encoder.py            文字単位の文脈エンコーダ・読みエンコーダ
├── vocab.py              文字語彙 / カナ語彙（EMPTY_READING_ID）
├── losses.py / train.py  margin loss・学習ループ・checkpoint
├── export_onnx.py        ONNX 化と INT8 量子化
├── runtime.py            ONNX Runtime 推論（読みキャッシュ・numpy の Viterbi）
├── benchmark.py          レイテンシ計測（文長ごと・内訳つき）
├── metrics.py / analysis.py   評価指標と誤りの 3 分類
├── benchmark_data.py     Joyo benchmark のロード
├── baselines.py          OpenJTalk の読み（アクセント記号の除去・未知文字の検出）
├── official_eval.py      benchmark の公式評価ツール（jkyb-eval）の呼び出し
├── config.py             学習設定（TrainConfig）
└── data/                 学習データ（生成・フィルタ・ルビ・監査・前処理）
scripts/                  CLI（辞書キャッシュ・評価・学習・書き出し・計測）
configs/                  学習設定（size-xs〜l など）
prompts/                  データ生成用サブエージェントへの指示
infra/                    クラウド GPU（vast.ai）の運用スクリプト
tests/                    pytest
```

---

## 9. 設計上の不変条件

変えるときは、この節と関係するテストも直すこと。

1. **辞書の読みは UniDic の `pron` と `kana` の両方を採る**（§3-1）
2. **カナ正規化は `kana.py` の単一実装を、辞書・学習データ・評価のすべてで共有する**（[R-10](pitfalls.md#r-10)）
3. **`NodeScorer` は差し替え可能なインターフェースとして保つ。** 単体ノードを受けるメソッドを作らない
4. **ノードスコアリングは必ずバッチ化する**（[R-12](pitfalls.md#r-12)）
5. **CRF の分子は積グラフ上の制約付き forward で計算する。** 全経路を列挙しない（§5-3）
6. **CRF / Viterbi は ONNX に含めず、ホスト側に置く**（§7-2）
7. **辞書データをリポジトリに入れない。** `.gitignore` の `/data/` は先頭のスラッシュが必須（外すとソースパッケージ `src/lattice_g2p/data/` まで無視される）
8. **読みの由来（`Entry.source` / `Node.source`）を保つ。** pron 形と kana 形は同じコストで区別できない
9. **`phonetic_key` は比較専用。** 長音の書き分け（コウ ≡ コー）と四つ仮名（チヂム ≡ チジム）を吸収する。出力する読みには使わない
10. **精度は文全体の読みで測る。** 対象語の span 単位の一致は参考値（[R-15](pitfalls.md#r-15)）
11. **空読みノードを 0 ベクトルにしない。** `EMPTY_READING_ID` を割り当て、全マスク行は `safe_mask` で避ける（[R-18](pitfalls.md#r-18)）
12. **dev の語は、どのデータ源からも学習に入れない。** 生成データの分割とルビ側の除外（`drop_held_out_spans`）の両方が要る
13. **部分制約では span の境界をまたぐノードを採用しない**（§5-6）
14. **遷移項を足しても積グラフは変えない。** DP を辺で回し、`transitions=None` では従来の実装をそのまま通す（§5-9）
15. **`Entry` にフィールドを足したら `_CACHE_FIELDS` と `from_entries` も直す**（[R-25](pitfalls.md#r-25)）
16. **対象語の読みを 1 本の経路から取り出さない**（§5-5）
17. **仮名・漢字を含む表記に空読みを許さない**（§3-2）
18. **連接行列を使うときは、連接 ID を持たないノードの扱いを明示し、品詞を保つ辞書を使う**（§4-8・[R-27](pitfalls.md#r-27)）
