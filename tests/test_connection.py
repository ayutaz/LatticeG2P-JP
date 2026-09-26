"""UniDic の連接行列（matrix.bin）の読み込みテスト（M7 Task 1）.

★軸の向きを小さな合成行列で固定する。軸を取り違えても値は「それらしく」
出るので、テストで向きを縛らないと気づけない（R-22 と同じ構図）。

MeCab の Connector::cost:
    cost(prev -> next) = matrix[prev.right_id + lsize * next.left_id]
"""

from pathlib import Path

import numpy as np
import pytest

from lattice_g2p.connection import BOS_EOS_ID, ConnectionMatrix, char_class, unk_ids


def _write_matrix(path: Path, lsize: int, rsize: int) -> None:
    """値 = 100 * prev_right_id + next_left_id にして、軸の取り違えを検出できるようにする."""
    m = np.zeros(lsize * rsize, dtype=np.int16)
    for next_left in range(rsize):
        for prev_right in range(lsize):
            m[prev_right + lsize * next_left] = 100 * prev_right + next_left
    with path.open("wb") as f:
        np.array([lsize, rsize], dtype=np.uint16).tofile(f)
        m.tofile(f)


# --- 読み込みと軸の向き ---


def test_cost_uses_prev_right_and_next_left(tmp_path: Path) -> None:
    """★cost(prev -> next) は prev の right_id と next の left_id で引く."""
    _write_matrix(tmp_path / "m.bin", lsize=5, rsize=3)
    cm = ConnectionMatrix.load(tmp_path / "m.bin")

    assert cm.cost(prev_right_id=4, next_left_id=2) == 402
    assert cm.cost(prev_right_id=2, next_left_id=0) == 200
    assert (cm.lsize, cm.rsize) == (5, 3)


def test_costs_is_the_vectorized_cost(tmp_path: Path) -> None:
    """一括版は (len(prev), len(next)) の行列を返し、cost と一致すること."""
    _write_matrix(tmp_path / "m.bin", lsize=5, rsize=3)
    cm = ConnectionMatrix.load(tmp_path / "m.bin")

    prev = np.array([0, 4, 2])
    nxt = np.array([2, 1])
    got = cm.costs(prev, nxt)

    assert got.shape == (3, 2)
    for i, p in enumerate(prev):
        for j, n in enumerate(nxt):
            assert got[i, j] == cm.cost(int(p), int(n))


def test_bos_and_eos_use_id_zero(tmp_path: Path) -> None:
    """BOS -> next は prev_right = 0、prev -> EOS は next_left = 0."""
    _write_matrix(tmp_path / "m.bin", lsize=5, rsize=3)
    cm = ConnectionMatrix.load(tmp_path / "m.bin")

    assert BOS_EOS_ID == 0
    assert list(cm.bos_costs(np.array([1, 2]))) == [1, 2]
    assert list(cm.eos_costs(np.array([3, 4]))) == [300, 400]


def test_rejects_ids_outside_the_matrix(tmp_path: Path) -> None:
    """★範囲外の ID は黙って読まない（隣の行を読んでしまう）."""
    _write_matrix(tmp_path / "m.bin", lsize=5, rsize=3)
    cm = ConnectionMatrix.load(tmp_path / "m.bin")

    with pytest.raises(IndexError):
        cm.cost(prev_right_id=5, next_left_id=0)
    with pytest.raises(IndexError):
        cm.cost(prev_right_id=0, next_left_id=3)
    with pytest.raises(IndexError):
        cm.costs(np.array([0]), np.array([3]))


def test_rejects_a_truncated_file(tmp_path: Path) -> None:
    """★サイズがヘッダと合わないファイルは読まない."""
    path = tmp_path / "m.bin"
    _write_matrix(path, lsize=5, rsize=3)
    path.write_bytes(path.read_bytes()[:-2])

    with pytest.raises(ValueError, match="サイズ"):
        ConnectionMatrix.load(path)


# --- 未知語・補完ノードの ID ---


UNK_DEF = """\
DEFAULT,537,4387,-1179,補助記号,一般,*,*,*,*
KANJI,14643,14497,10330,名詞,普通名詞,一般,*,*,*
KANJI,10006,850,12831,名詞,普通名詞,サ変可能,*,*,*
HIRAGANA,6367,2250,10147,感動詞,一般,*,*,*,*
HIRAGANA,14643,14497,11197,名詞,普通名詞,一般,*,*,*
"""


def test_unk_ids_pick_the_cheapest_entry_per_class(tmp_path: Path) -> None:
    """★文字種ごとに最小コストの候補を採る（文脈が無いときに MeCab が選ぶもの）."""
    path = tmp_path / "unk.def"
    path.write_text(UNK_DEF, encoding="utf-8")

    ids = unk_ids(path)

    assert ids["KANJI"] == (14643, 14497)
    assert ids["HIRAGANA"] == (6367, 2250)  # 10147 < 11197
    assert ids["DEFAULT"] == (537, 4387)


@pytest.mark.parametrize(
    ("ch", "expected"),
    [
        ("米", "KANJI"),
        ("三", "KANJINUMERIC"),
        ("あ", "HIRAGANA"),
        ("ア", "KATAKANA"),
        ("7", "NUMERIC"),
        ("７", "NUMERIC"),
        ("A", "ALPHA"),
        ("。", "DEFAULT"),
    ],
)
def test_char_class(ch: str, expected: str) -> None:
    assert char_class(ch) == expected
