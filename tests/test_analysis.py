from lattice_g2p.analysis import ErrorCategory, classify_error, summarize
from lattice_g2p.dictionary import Dictionary, Entry


def _dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0),
            Entry("米", "ベイ", "名詞", 1),
            Entry("国", "コク", "名詞", 2),
        ]
    )


def test_path_selection_error_when_gold_is_in_lattice() -> None:
    """正解が辞書にあるのに選べなかった → モデルの問題。"""
    assert classify_error("米国", "コメコク", _dic()) is ErrorCategory.PATH_SELECTION


def test_oracle_failure_when_gold_not_producible() -> None:
    """正解が辞書で作れない → 辞書の問題。モデルを直しても改善しない。"""
    assert classify_error("米国", "イリガビ", _dic()) is ErrorCategory.ORACLE_FAILURE


def test_normalization_error_when_gold_is_not_normalized() -> None:
    """正解読みが正規化されていない → バグ。"""
    assert classify_error("米国", "こめこく", _dic()) is ErrorCategory.NORMALIZATION


def test_summarize_counts_all_categories() -> None:
    counts = summarize([ErrorCategory.PATH_SELECTION, ErrorCategory.PATH_SELECTION])

    assert counts[ErrorCategory.PATH_SELECTION] == 2
    assert counts[ErrorCategory.ORACLE_FAILURE] == 0
    assert set(counts) == set(ErrorCategory)


def test_summarize_empty() -> None:
    assert all(v == 0 for v in summarize([]).values())
