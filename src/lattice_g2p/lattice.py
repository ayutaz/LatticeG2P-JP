"""単語ラティス (DAG) の構築.

頂点は文字位置 0..L, 辺は Node (表記 + 読み).
★同じ表記でも読みが異なれば別ノードになる.

すべての位置に 1 文字ノードを用意することで, 位置 0 から L への経路を必ず確保する.
経路が途切れると Oracle 判定も CRF の forward も成立しない.
"""

from dataclasses import dataclass, field

from lattice_g2p.dictionary import Dictionary, allows_empty_reading
from lattice_g2p.kana import is_kana_only, normalize_kana

_HIRAGANA_START = "ぁ"
_HIRAGANA_END = "ゖ"
_KATAKANA_START = "ァ"
_HIRA_TO_KATA_OFFSET = ord(_KATAKANA_START) - ord(_HIRAGANA_START)

UNK_READING = "〇"
"""〇. 読みが決まらない 1 文字のマーカ.

空文字にしないこと. 空読みにすると, その文字を読み飛ばした経路が正解になりうる.
"""

FALLBACK_COST = 10000
"""フォールバックノードの生起コスト. 辞書エントリより明確に不利にする."""


@dataclass(frozen=True)
class Node:
    """ラティスの辺. text[start:end] を reading と読む候補."""

    start: int
    end: int
    surface: str
    reading: str
    entry_id: int
    pos: str
    cost: int = 0
    source: str = ""
    """読みの由来 ("pron" / "kana" / "both" / "aug"). dictionary.Entry と同じ."""

    left_id: int = 0
    right_id: int = 0
    """連接 ID. dictionary.Entry と同じ (M6). 未知語ノードは 0."""


@dataclass
class Lattice:
    text: str
    nodes: list[Node]
    nodes_starting_at: list[list[int]] = field(default_factory=list)
    nodes_ending_at: list[list[int]] = field(default_factory=list)

    def has_full_path(self) -> bool:
        """位置 0 から len(text) への経路が存在するか."""
        n = len(self.text)
        reachable = [False] * (n + 1)
        reachable[0] = True
        for i in range(n):
            if not reachable[i]:
                continue
            for ni in self.nodes_starting_at[i]:
                reachable[self.nodes[ni].end] = True
        return reachable[n]


def _fallback_readings(ch: str, dic: Dictionary) -> list[str]:
    """1 文字に対するフォールバック読みを返す."""
    if _HIRAGANA_START <= ch <= _HIRAGANA_END:
        return [chr(ord(ch) + _HIRA_TO_KATA_OFFSET)]

    normalized = normalize_kana(ch)
    if normalized and is_kana_only(normalized):
        return [normalized]

    readings = dic.readings_of(ch)
    if readings:
        return readings

    if allows_empty_reading(ch):
        return [""]
    return [UNK_READING]


def build_lattice(text: str, dic: Dictionary, max_node_length: int | None = None) -> Lattice:
    """テキストから単語ラティスを構築する.

    Args:
        max_node_length: これより長いエントリを採用しない. None なら制限なし.
            ノード数が爆発する場合に絞るためのつまみ.
    """
    n = len(text)
    seen_reading: set[tuple[int, int, str, str]] = set()
    seen_full: set[tuple[int, int, str, str, int, int]] = set()
    nodes: list[Node] = []

    def add(node: Node, pos_variants: bool) -> None:
        """重複を除いて足す.

        辞書ノードは連接 ID まで含めて比べる（pos_variants=True）. 品詞を保つ辞書
        （keep_pos）では、同じ span・同じ読みでも品詞違いが別ノードになる（R-27）.
        ★既定の辞書では (表記, 読み) が一意なので、従来と同じノード列になる.
        フォールバックは読みで比べる. 同じ読みの辞書ノードがあれば立てない.
        """
        key = (node.start, node.end, node.surface, node.reading)
        full = (*key, node.left_id, node.right_id)
        if (full in seen_full) if pos_variants else (key in seen_reading):
            return
        seen_reading.add(key)
        seen_full.add(full)
        nodes.append(node)

    for i in range(n):
        for entry in dic.common_prefix_search(text, i):
            length = len(entry.surface)
            if max_node_length is not None and length > max_node_length:
                continue
            add(
                Node(
                    start=i,
                    end=i + length,
                    surface=entry.surface,
                    reading=entry.reading,
                    entry_id=entry.entry_id,
                    pos=entry.pos,
                    cost=entry.cost,
                    source=entry.source,
                    left_id=entry.left_id,
                    right_id=entry.right_id,
                ),
                pos_variants=True,
            )

    # 未知語フォールバック: すべての位置に 1 文字ノードを置き, 経路を必ず確保する
    for i in range(n):
        ch = text[i]
        for reading in _fallback_readings(ch, dic):
            add(
                Node(
                    start=i,
                    end=i + 1,
                    surface=ch,
                    reading=reading,
                    entry_id=-1,
                    pos="UNK",
                    cost=FALLBACK_COST,
                ),
                pos_variants=False,
            )

    starting: list[list[int]] = [[] for _ in range(n + 1)]
    ending: list[list[int]] = [[] for _ in range(n + 1)]
    for idx, node in enumerate(nodes):
        starting[node.start].append(idx)
        ending[node.end].append(idx)

    return Lattice(text=text, nodes=nodes, nodes_starting_at=starting, nodes_ending_at=ending)
