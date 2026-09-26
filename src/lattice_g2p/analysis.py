"""誤り分析.

★この 3 分類を出さない評価は無意味である (docs/evaluation.md §6-3).
  Accuracy が 98% のとき, 残り 2% がどれに属するかで次のアクションが全く変わる.

  Oracle 失敗    -> 辞書の問題. モデルを改善しても直らない
  経路選択の誤り  -> モデルの問題. ここが改善対象
  正規化の不一致  -> バグ
"""

from collections import Counter
from enum import Enum

from lattice_g2p.crf import build_constrained_graph
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import normalize_kana
from lattice_g2p.lattice import build_lattice


class ErrorCategory(Enum):
    ORACLE_FAILURE = "Oracle 失敗（辞書に正解がない。辞書の問題）"
    PATH_SELECTION = "経路選択の誤り（正解は辞書にある。モデルの問題）"
    NORMALIZATION = "正規化の不一致（バグ）"


def classify_error(text: str, gold_reading: str, dic: Dictionary) -> ErrorCategory:
    """誤りを 3 分類する.

    Args:
        gold_reading: 文全体の正解読み (正規化済みであるべき)
    """
    normalized = normalize_kana(gold_reading)
    if normalized != gold_reading:
        return ErrorCategory.NORMALIZATION

    lattice = build_lattice(text, dic)
    if build_constrained_graph(lattice, normalized).is_reachable():
        return ErrorCategory.PATH_SELECTION
    return ErrorCategory.ORACLE_FAILURE


def summarize(errors: list[ErrorCategory]) -> dict[ErrorCategory, int]:
    counts = Counter(errors)
    return {c: counts.get(c, 0) for c in ErrorCategory}
