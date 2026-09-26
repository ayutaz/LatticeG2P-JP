"""UniDic の連接行列（MeCab の matrix.bin）を読む（M7）.

★M6 で CRF に遷移項を足したが、試したのは連接行列の**低ランク近似**
（16 / 32 次元の内積）であって行列そのものではなかった。M7 では行列そのものを
固定の遷移スコアとして差し込み、「遷移項で差が埋まるか」の上限を測る
（docs/results.md §3）。

形式（MeCab の Connector）:

    先頭 4 バイト   uint16 lsize, uint16 rsize
    残り            int16 が lsize * rsize 個

    cost(prev -> next) = matrix[prev.right_id + lsize * next.left_id]

★行優先で (rsize, lsize) に reshape すると table[next.left_id, prev.right_id] になる。
**軸を取り違えても値は「それらしく」出る**ので、tests/test_connection.py で向きを縛り、
scripts/verify_matrix.py で実ファイルの matrix.def（テキスト）と突き合わせる。

UniDic CWJ 3.1.1: lsize = 15,626 / rsize = 15,388。int16 で 480,905,780 バイト。
★np.memmap で開く。480 MB を読み込まない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lattice_g2p.lattice import Node

BOS_EOS_ID = 0
"""BOS / EOS の連接 ID. matrix.def の 1 行目 `0 0 5791` は BOS -> EOS."""

_HEADER_BYTES = 4


class ConnectionMatrix:
    """連接コスト表. 値は MeCab の**コスト**（小さいほど繋がりやすい）."""

    def __init__(self, table: np.ndarray) -> None:
        self.table = table
        """(rsize, lsize). table[next_left_id, prev_right_id]."""
        self.rsize, self.lsize = table.shape

    @classmethod
    def load(cls, path: Path) -> ConnectionMatrix:
        path = Path(path)
        with path.open("rb") as f:
            lsize, rsize = (int(x) for x in np.fromfile(f, dtype=np.uint16, count=2))
        expected = _HEADER_BYTES + 2 * lsize * rsize
        actual = path.stat().st_size
        if actual != expected:
            raise ValueError(
                f"{path} のサイズがヘッダと合いません: {actual:,} バイト"
                f"（lsize={lsize} × rsize={rsize} なら {expected:,} バイト）。"
                "壊れているか、matrix.bin ではありません。"
            )
        table = np.memmap(
            path, dtype=np.int16, mode="r", offset=_HEADER_BYTES, shape=(rsize, lsize)
        )
        return cls(table)

    # --- 参照 ---

    def _check(self, prev_right_ids: np.ndarray, next_left_ids: np.ndarray) -> None:
        """★範囲外の ID は黙って読まない. 隣の行を読んでも値はそれらしく見える."""
        if prev_right_ids.size and (
            prev_right_ids.min() < 0 or prev_right_ids.max() >= self.lsize
        ):
            raise IndexError(
                f"right_id が範囲外です（0 <= id < {self.lsize}）: "
                f"{prev_right_ids.min()}〜{prev_right_ids.max()}"
            )
        if next_left_ids.size and (
            next_left_ids.min() < 0 or next_left_ids.max() >= self.rsize
        ):
            raise IndexError(
                f"left_id が範囲外です（0 <= id < {self.rsize}）: "
                f"{next_left_ids.min()}〜{next_left_ids.max()}"
            )

    def cost(self, prev_right_id: int, next_left_id: int) -> int:
        self._check(np.array([prev_right_id]), np.array([next_left_id]))
        return int(self.table[next_left_id, prev_right_id])

    def costs(self, prev_right_ids: np.ndarray, next_left_ids: np.ndarray) -> np.ndarray:
        """(len(prev), len(next)) の連接コスト. [i, j] = cost(prev[i] -> next[j])."""
        prev = np.asarray(prev_right_ids, dtype=np.int64)
        nxt = np.asarray(next_left_ids, dtype=np.int64)
        self._check(prev, nxt)
        return np.asarray(self.table[np.ix_(nxt, prev)], dtype=np.int32).T

    def bos_costs(self, next_left_ids: np.ndarray) -> np.ndarray:
        """BOS -> 各ノード."""
        return self.costs(np.array([BOS_EOS_ID]), next_left_ids)[0]

    def eos_costs(self, prev_right_ids: np.ndarray) -> np.ndarray:
        """各ノード -> EOS."""
        return self.costs(prev_right_ids, np.array([BOS_EOS_ID]))[:, 0]


# --- 未知語・補完ノードの ID ---


def unk_ids(path: Path) -> dict[str, tuple[int, int]]:
    """unk.def から、文字種 -> (left_id, right_id) を作る.

    ★文字種ごとに**最小コスト**の候補を採る. 文脈が無いときに MeCab が選ぶもの.
    UniDic 3.1.1 では、どの文字種でも名詞-普通名詞-一般が最小だった
    （KANJI -> 14643 / 14497）。
    """
    best: dict[str, tuple[int, int, int]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, left, right, cost = line.split(",")[:4]
        candidate = (int(cost), int(left), int(right))
        if cls not in best or candidate < best[cls]:
            best[cls] = candidate
    return {cls: (left, right) for cls, (_, left, right) in best.items()}


_KANJI_NUMERIC = set("〇一二三四五六七八九十百千万億兆")


def char_class(ch: str) -> str:
    """1 文字の文字種. unk.def の分類名に合わせる（char.def の簡略版）."""
    if ch in _KANJI_NUMERIC:
        return "KANJINUMERIC"
    o = ord(ch)
    if ch == "々" or 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
        return "KANJI"
    if 0x3041 <= o <= 0x309F:
        return "HIRAGANA"
    if 0x30A0 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "KATAKANA"
    if ch.isdigit():
        return "NUMERIC"
    if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("Ａ" <= ch <= "Ｚ") or ("ａ" <= ch <= "ｚ"):
        return "ALPHA"
    if 0x0370 <= o <= 0x03FF:
        return "GREEK"
    if 0x0400 <= o <= 0x04FF:
        return "CYRILLIC"
    if ch.isspace():
        return "SPACE"
    return "DEFAULT"


def resolve_ids(
    nodes: list[Node], unk: dict[str, tuple[int, int]] | None
) -> tuple[np.ndarray, np.ndarray]:
    """各ノードの (left_ids, right_ids).

    ★補完ノード（source="aug"）と未知語フォールバック（entry_id < 0）は ID が 0 で、
    **BOS / EOS の ID と衝突する**。unk が与えられれば文字種の ID に置き換える
    （MeCab が辞書に無い文字に与えるもの）。None なら置き換えない（呼び出し側で
    遷移を 0 にする感度分析用）.
    """
    left = np.empty(len(nodes), dtype=np.int64)
    right = np.empty(len(nodes), dtype=np.int64)
    for i, n in enumerate(nodes):
        if unk is not None and (n.source == "aug" or n.entry_id < 0):
            cls = char_class(n.surface[0]) if n.surface else "DEFAULT"
            left[i], right[i] = unk.get(cls, unk["DEFAULT"])
        else:
            left[i], right[i] = n.left_id, n.right_id
    return left, right


def is_unresolved(node: Node) -> bool:
    """連接 ID を辞書から持たないノード（補完・未知語）."""
    return node.source == "aug" or node.entry_id < 0
