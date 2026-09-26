"""生成データのフィルタリング.

★ラティス整合チェックは品質管理ではなく動作要件である.
  正解経路が存在しない例は CRF の分子が計算できず, 学習が壊れる
  (constrained_logsumexp が ValueError を投げる).

★落ちた例は捨てるだけでなく必ず集計する.
  ラティス整合の通過率が低い場合, LLM の品質ではなく辞書かコードの問題を
  先に疑う (docs/training.md §3).
"""

import re
from dataclasses import dataclass, field

from lattice_g2p.crf import (
    ConstrainedGraph,
    build_constrained_graph,
    covering_kana_windows,
    target_readings_on_gold_paths,
)
from lattice_g2p.data.schema import GeneratedSentence, TrainingExample
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import is_kana_only, normalize_kana, phonetic_key
from lattice_g2p.lattice import Lattice, build_lattice

DEFAULT_MIN_LEN = 5
DEFAULT_MAX_LEN = 80

OUT_OF_SCOPE = re.compile(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]")
"""範囲外の文字種 (docs/pitfalls.md R-05). 前段の Text Normalizer の担当."""


@dataclass
class FilterStats:
    """各段の通過数. 歩留まりの記録は品質管理の要."""

    generated: int = 0
    format_ok: int = 0
    lattice_ok: int = 0
    target_ok: int = 0
    dedup_ok: int = 0
    lattice_failures: list[str] = field(default_factory=list)

    def format(self) -> str:
        def pct(n: int, d: int) -> str:
            return f"{n / d * 100:5.1f}%" if d else "  n/a"

        return (
            f"生成           {self.generated:7,}\n"
            f"├ 形式         {self.format_ok:7,}  ({pct(self.format_ok, self.generated)})\n"
            f"├ ラティス整合  {self.lattice_ok:7,}  ({pct(self.lattice_ok, self.format_ok)})"
            f"   ★95% を下回るなら辞書かコードの問題\n"
            f"├ 助詞+対象語   {self.target_ok:7,}  ({pct(self.target_ok, self.lattice_ok)})\n"
            f"└ 重複除去      {self.dedup_ok:7,}  ({pct(self.dedup_ok, self.target_ok)})\n"
            f"最終歩留まり    {pct(self.dedup_ok, self.generated)}"
        )


def _format_ok(row: GeneratedSentence, min_len: int, max_len: int) -> bool:
    reading = normalize_kana(row.reading)
    if not reading or not is_kana_only(reading):
        return False
    if not (min_len <= len(row.text) <= max_len):
        return False
    if OUT_OF_SCOPE.search(row.text):
        return False
    if not (0 <= row.target_start < row.target_end <= len(row.text)):
        return False
    return row.text[row.target_start : row.target_end] == row.target_surface


def any_gold_path(graph: ConstrainedGraph) -> list[int]:
    """正解経路を 1 本だけ取り出す (ノード index の列).

    枝刈り済みのグラフなので, どの辺を辿っても goal に到達できる.
    """
    if graph.goal_state is None:
        raise ValueError("正解経路が存在しません")

    out: dict[int, tuple[int, int]] = {}
    for node_idx, src, dst in graph.edges:
        out.setdefault(src, (node_idx, dst))

    path: list[int] = []
    state = graph.start_state
    while state != graph.goal_state:
        node_idx, nxt = out[state]
        path.append(node_idx)
        state = nxt
    return path


def _span_reading_on_path(lattice: Lattice, path: list[int], start: int, end: int) -> str:
    return "".join(
        lattice.nodes[i].reading
        for i in path
        if lattice.nodes[i].start < end and lattice.nodes[i].end > start
    )


PARTICLE_READINGS = {"は": "ワ", "へ": "エ"}
"""助詞として使われるときの正しい読み.

★UniDic には助詞「は」の pron=ハ エントリも (低頻度で) 存在するため,
ラティス整合チェックでは「私はガクセイ」を ワタシ**ハ**ガクセイ と読む文が
通ってしまう.

助詞「は」は日本語で最頻出の語であり, これが学習データに混ざると
助詞の読みを誤って学習する. 実測で生成モデルがこの誤りを出したため,
機械的に弾く.
"""


def _particle_readings_ok(graph: ConstrainedGraph, lattice: Lattice) -> bool:
    """助詞の「は」「へ」を誤って読む経路しか無い場合に False を返す.

    ★表記が「は」「へ」で品詞が助詞のノードだけを見る.
    「葉」のように ハ と読む名詞は対象外.

    ★「正解経路のどれかに含まれる」で落としてはいけない.
    「はがき」のような語頭が「は」の語では, ラティスに
    は(助詞, pron=ハ) + がき という経路も立つ. それを理由に落とすと
    有効なデータが消える (M3d の生成データで実測).

    落とすべきなのは**避けられない**場合だけである. 「私は学生」を
    ワタシハガクセイ と読む文は, 助詞「は」を ハ と読む以外に
    その読みを作れない.
    """
    bad = {
        i
        for i, node in enumerate(lattice.nodes)
        if PARTICLE_READINGS.get(node.surface) is not None
        and node.pos.startswith("助詞")
        and node.reading != PARTICLE_READINGS[node.surface]
    }
    if not bad:
        return True
    return _goal_reachable_without(graph, bad)


def _goal_reachable_without(graph: ConstrainedGraph, excluded: set[int]) -> bool:
    """指定したノードを使わずに goal へ到達できる正解経路があるか."""
    if graph.goal_state is None:
        return False

    out: dict[int, list[int]] = {}
    for node_idx, src, dst in graph.edges:
        if node_idx in excluded:
            continue
        out.setdefault(src, []).append(dst)

    seen = {graph.start_state}
    stack = [graph.start_state]
    while stack:
        state = stack.pop()
        if state == graph.goal_state:
            return True
        for nxt in out.get(state, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _target_reading_ok(
    graph: ConstrainedGraph,
    lattice: Lattice,
    gold_reading: str,
    expected: str,
    target_start: int,
    target_end: int,
) -> bool:
    """対象語が, 指定された読みで使われているか.

    このフィルタの目的は「LLM が指定した読みを無視していないか」の確認である.
    CRF は文全体の読みで学習するので, 対象語の読みが分離できなくても
    学習データとしては有効であることに注意.

    判定は 2 段階:

    1. 正解経路のどれかが span の境界を持つなら, その読みと一致するかを見る (厳密)
    2. どの経路も境界を持たないなら (「固める -> カタメル」で target が「固」だけ,
       など), 対象語を覆うノードのカナ範囲の中に読みが現れるかを見る (緩い).
       文全体から探すと「コ」が「ココロ」に一致してしまい, 緩すぎる.

    ★1 だけにすると歩留まりが 24 ポイント落ちる. 対象語が長いノードの内側に
    入るケースが実データで珍しくないため (docs/training.md §3).
    """
    expected_key = phonetic_key(expected)

    candidates = target_readings_on_gold_paths(graph, gold_reading, target_start, target_end)
    if candidates:
        return any(phonetic_key(c) == expected_key for c in candidates)

    windows = covering_kana_windows(graph, lattice, target_start, target_end)
    return any(
        expected_key in phonetic_key(gold_reading[k1:k2]) for k1, k2 in windows
    )


def filter_sentences(
    rows: list[GeneratedSentence],
    dic: Dictionary,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
) -> tuple[list[TrainingExample], FilterStats]:
    """生成データを 4 段のフィルタに通す.

    Returns:
        (学習に使える例, 各段の通過数)
    """
    stats = FilterStats(generated=len(rows))
    kept: list[TrainingExample] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        # [1] 形式チェック
        if not _format_ok(row, min_len, max_len):
            continue
        stats.format_ok += 1

        # [2] ★ラティス整合チェック (= Oracle チェック)
        gold = normalize_kana(row.reading)
        lattice = build_lattice(row.text, dic)
        graph = build_constrained_graph(lattice, gold)
        if not graph.is_reachable():
            stats.lattice_failures.append(f"{row.text} / {gold}")
            continue
        stats.lattice_ok += 1

        # [3a] 助詞の読みが正しいか
        if not _particle_readings_ok(graph, lattice):
            continue

        # [3b] 対象語の読みが意図どおりか
        expected = normalize_kana(row.target_reading)
        if not _target_reading_ok(
            graph, lattice, gold, expected, row.target_start, row.target_end
        ):
            continue
        stats.target_ok += 1

        # [4] 重複除去
        key = (row.text, gold)
        if key in seen:
            continue
        seen.add(key)
        stats.dedup_ok += 1

        kept.append(
            TrainingExample(
                text=row.text,
                reading=gold,
                target_start=row.target_start,
                target_end=row.target_end,
                target_surface=row.target_surface,
                target_reading=expected,
                meta={
                    "prompt_version": row.prompt_version,
                    "model": row.model,
                    "generated_at": row.generated_at,
                },
            )
        )

    return kept, stats


def drop_held_out_chars(rows: list, chars: set[str]) -> tuple[list, int]:  # noqa: ANN001
    """指定した文字を本文に含む例を落とす.

    ★生成データは文全体の読みが教師なので, dev に取り分けた文字が本文に
    出てくるだけでその読みが教えられてしまう (docs/pitfalls.md R-20).

    ルビ由来のデータとは条件が違う. あちらは span の外を CRF が周辺化するので,
    本文に出るだけならラベルは付かない (data/ruby.py の drop_held_out_spans).

    M3d で 数詞 + 助数詞 のデータを作るときに使う. dev は 四・六・八 を
    held-out にしているので, それらを含む文は学習に使えない.
    """
    if not chars:
        return rows, 0
    kept = [row for row in rows if not any(ch in row.text for ch in chars)]
    return kept, len(rows) - len(kept)
