"""青空文庫のルビから学習データを取り出す.

★ルビは「実際の出版物で人間が付けた読み」なので hallucination がない.
LLM 生成データの最大の弱点 (フィルタで検出できない文脈的な誤読) がそもそも無い.

制約は 2 つある.

1. ルビは文の一部にしか付かない
   -> 文全体の読みを前提とする build_constrained_graph では扱えない.
      crf.build_partially_constrained_graph を使う.

2. 義訓ルビ (運命《さだめ》) と歴史的仮名遣い (今日《けふ》) が混ざる
   -> どちらも辞書に無い読みなので, ラティス整合チェックが自動的に落とす.
      辞書制約付きという設計がそのまま品質フィルタになっている.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from lattice_g2p.crf import build_partially_constrained_graph
from lattice_g2p.data.schema import read_jsonl
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import normalize_kana
from lattice_g2p.lattice import Lattice, build_lattice

RUBY_OPEN = "《"
RUBY_CLOSE = "》"
RUBY_BAR = "｜"
"""全角縦棒. ルビを掛ける範囲の開始を明示する記号."""

NOTE_CLOSE = "］"
_NOTE_OPEN = "［＃"
"""入力者注 (［＃「米」に傍点］ など). 本文ではないので落とす."""

_SENTENCE_ENDERS = "。！？"
_QUOTE_BRACKETS = "「」『』"

_KANJI_RANGES = (
    (0x3005, 0x3007),  # 々 〆 〇
    (0x30F6, 0x30F6),  # ヶ (一ヶ月)
    (0x3400, 0x4DBF),  # CJK 拡張 A
    (0x4E00, 0x9FFF),  # CJK 統合漢字
    (0xF900, 0xFAFF),  # CJK 互換漢字
)

_IN_SCOPE = re.compile(r"^[ぁ-ゖァ-ヺーゝゞ々〆〇㐀-䶿一-鿿豈-﫿、。！？]+$")
"""第 1 マイルストーンの範囲内の文字だけからなるか (docs/pitfalls.md R-05).

数字・英字・括弧・三点リーダなどを含む文は落とす.
"""

DEFAULT_MIN_LEN = 6
DEFAULT_MAX_LEN = 60


@dataclass(frozen=True)
class RubyExample:
    """ルビから取り出した学習データ 1 件.

    ★文全体の読みは持たない. spans の位置だけ読みが分かっている.
    """

    text: str
    spans: list[tuple[int, int, str]] = field(default_factory=list)
    """(開始文字位置, 終了文字位置, 読み). 互いに素."""

    source: str = ""
    """出典 (青空文庫の作品 ID など). 追跡できるようにしておく."""


def _is_kanji(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _KANJI_RANGES)


def _kanji_run_start(plain: list[str]) -> int | None:
    """末尾から漢字の連続をさかのぼった開始位置. 漢字で終わっていなければ None."""
    i = len(plain)
    while i > 0 and _is_kanji(plain[i - 1]):
        i -= 1
    return i if i < len(plain) else None


def extract_ruby(line: str) -> tuple[str, list[tuple[int, int, str]]]:
    """1 行からルビ記法を取り除き, 本文と読みの span を返す.

    ルビの掛かる範囲は次の規則で決める.

        ｜生《なま》ビール   ->  ｜ から 《 の直前まで
        生真面目《きまじめ》 ->  直前の漢字の連続

    ★どちらでも決まらない場合 (直前が仮名で ｜ も無い) はそのルビを捨てる.
    どこに掛かるか推測しない. 誤った位置に読みを付けると, ルビを使う唯一の
    利点 (人間が付けた正確さ) が失われる.

    前提: ｜ の後には必ず 《...》 が来る. もし来なければ, 次のルビが
    その ｜ の位置から掛かってしまう (基底が長くなりすぎる). 実データで
    確認したところ 398 作品・｜ 3,158 件すべてで後続にルビがあり,
    この破綻は起きなかった. 破綻を検出したらここに guard を足すこと.

    Returns:
        (ルビ記法を除いた本文, [(開始, 終了, カタカナ読み), ...])
    """
    plain: list[str] = []
    spans: list[tuple[int, int, str]] = []
    bar: int | None = None
    i = 0
    n = len(line)

    while i < n:
        ch = line[i]

        if line.startswith(_NOTE_OPEN, i):
            close = line.find(NOTE_CLOSE, i)
            i = i + 1 if close == -1 else close + 1
            continue

        if ch == RUBY_BAR:
            bar = len(plain)
            i += 1
            continue

        if ch == RUBY_OPEN:
            close = line.find(RUBY_CLOSE, i)
            if close == -1:
                i += 1  # 壊れた行. 《 だけ捨てて先へ進む
                continue
            reading = normalize_kana(line[i + 1 : close])
            i = close + 1
            start = bar if bar is not None else _kanji_run_start(plain)
            bar = None
            end = len(plain)
            if start is not None and start < end and reading:
                spans.append((start, end, reading))
            continue

        plain.append(ch)
        i += 1

    return "".join(plain), spans


_FENCE = re.compile(r"^-{10,}$")


def extract_body(raw: str) -> str:
    """青空文庫のテキストからヘッダ (書誌・凡例) と奥付を落として本文を返す.

    凡例は 2 本の罫線で囲まれている. 奥付は「底本：」から始まる.
    """
    lines = raw.replace("\r\n", "\n").split("\n")

    fences = [i for i, line in enumerate(lines) if _FENCE.match(line.strip())]
    if len(fences) >= 2:
        lines = lines[fences[1] + 1 :]

    body: list[str] = []
    for line in lines:
        if line.startswith("底本"):
            break
        body.append(line)
    return "\n".join(body).strip()


def iter_sentences(body: str) -> Iterator[str]:
    """本文を文に分割する.

    ★文末記号で終わらない断片は捨てる. 見出しや行の途中で切れたものを
    学習データにすると, 不自然な文脈を教えることになる.
    """
    buf: list[str] = []
    for ch in body:
        if ch in _QUOTE_BRACKETS:
            continue
        if ch == "\n":
            buf = []
            continue
        buf.append(ch)
        if ch in _SENTENCE_ENDERS:
            yield "".join(buf)
            buf = []


def is_in_scope(text: str) -> bool:
    """第 1 マイルストーンの範囲内の文字だけか (docs/pitfalls.md R-05)."""
    return bool(_IN_SCOPE.match(text))


def iter_ruby_examples(
    body: str,
    source: str = "",
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> Iterator[RubyExample]:
    """本文からルビ付きの文だけを取り出す.

    ここでは辞書を見ない. ラティス整合は呼び出し側 (filter_ruby_examples) で行う.
    """
    for sentence in iter_sentences(body):
        text, spans = extract_ruby(sentence)
        if not spans:
            continue
        if not (min_len <= len(text) <= max_len):
            continue
        if not is_in_scope(text):
            continue
        yield RubyExample(text=text, spans=spans, source=source)


# --- ラティス整合フィルタ ---
#
# ★ここだけ辞書に依存する. 読みが辞書ラティスで生成できない span を落とす.
# M2 の filter_sentences と役割は同じだが, 判定の単位が「文」ではなく「span」.

@dataclass
class RubyFilterStats:
    total: int = 0
    kept: int = 0
    duplicates: int = 0
    spans_total: int = 0
    spans_kept: int = 0

    def format(self) -> str:
        span_rate = self.spans_kept / self.spans_total * 100 if self.spans_total else 0.0
        rate = self.kept / self.total * 100 if self.total else 0.0
        return "\n".join(
            [
                f"ルビ  {self.total:,} 文 -> {self.kept:,} 文 ({rate:.1f}%)",
                f"span  {self.spans_total:,} -> {self.spans_kept:,} ({span_rate:.1f}%)",
                f"重複除去  {self.duplicates:,} 件",
            ]
        )


def usable_spans(
    lattice: Lattice, spans: list[tuple[int, int, str]]
) -> list[tuple[int, int, str]]:
    """辞書ラティスで検証できる span だけを返す.

    ★span ごとに判定する. 1 つ落ちても残りは使える.
    文全体の読みを要求する M2 のフィルタでは, 1 箇所でも駄目なら文ごと捨てるしかない.

    落ちるのは主に次の 2 種類で, どちらも辞書に無い読みである.

        義訓ルビ           運命《さだめ》   著者独自の読み
        歴史的仮名遣い     今日《けふ》     旧仮名
    """
    kept = [
        span
        for span in spans
        if build_partially_constrained_graph(lattice, [span]).is_reachable()
    ]
    if kept and not build_partially_constrained_graph(lattice, kept).is_reachable():
        # 互いに素な span が個別に成立するなら同時にも成立するはずだが,
        # 念のため確認する. 破れたら安全側に倒して 1 つだけ残す.
        return kept[:1]
    return kept


def filter_ruby_examples(
    examples: list[RubyExample], dic: Dictionary
) -> tuple[list[RubyExample], RubyFilterStats]:
    """ラティス整合しない span を落とし, 使える span が無くなった文を捨てる."""
    stats = RubyFilterStats(total=len(examples))
    seen: set[tuple[str, tuple[tuple[int, int, str], ...]]] = set()
    kept: list[RubyExample] = []

    for example in examples:
        stats.spans_total += len(example.spans)
        lattice = build_lattice(example.text, dic)
        spans = usable_spans(lattice, example.spans)
        if not spans:
            continue
        stats.spans_kept += len(spans)

        key = (example.text, tuple(spans))
        if key in seen:
            stats.duplicates += 1
            continue
        seen.add(key)
        kept.append(RubyExample(text=example.text, spans=spans, source=example.source))

    stats.kept = len(kept)
    return kept, stats


def read_ruby_jsonl(path: Path) -> list[RubyExample]:
    """ルビ学習データを JSONL から読む.

    ★JSON はタプルを保てないので, span をリストからタプルに戻す.
    build_partially_constrained_graph は (start, end, reading) の並びで
    ソート・比較するため, 型が揺れると分かりにくい不具合になる.
    """
    return [
        RubyExample(
            text=row["text"],
            spans=[(int(s), int(e), r) for s, e, r in row.get("spans", [])],
            source=row.get("source", ""),
        )
        for row in read_jsonl(path)
    ]


def drop_held_out_spans(
    examples: list[RubyExample], held_out: set[str]
) -> tuple[list[RubyExample], int]:
    """dev に回した語のルビを落とす.

    ★dev はエントリ単位で分離している (docs/pitfalls.md R-09).
    ルビ側でその原則が破れると, dev の語の読みを学習時に教えてしまい,
    dev の精度が過大評価される. 「ルビを足したら伸びた」という比較が
    成立しなくなる.

    落とすのは **ルビが付いた span だけ**. 本文に出てくるだけなら残す.
    span の外は CRF が周辺化するのでラベルは付かず, リークにならない.
    ここまで落とすとルビの 25% が消えて割に合わない (実測).

    Returns:
        (残った例, 落とした span の数)
    """
    kept: list[RubyExample] = []
    dropped = 0
    for example in examples:
        spans = [s for s in example.spans if example.text[s[0] : s[1]] not in held_out]
        dropped += len(example.spans) - len(spans)
        if not spans:
            continue
        kept.append(RubyExample(text=example.text, spans=spans, source=example.source))
    return kept, dropped
