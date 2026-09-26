import pytest

from lattice_g2p.metrics import EvalResult, accuracy, accuracy_any, corpus_per, levenshtein


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("コメ", "コメ", 0),
        ("コメ", "ベイ", 2),
        ("コメ", "コメノ", 1),  # 挿入
        ("コメノ", "コメ", 1),  # 削除
        ("イナビカリ", "イリガビ", 4),
        ("", "コメ", 2),
        ("コメ", "", 2),
        ("", "", 0),
    ],
)
def test_levenshtein(a: str, b: str, expected: int) -> None:
    assert levenshtein(a, b) == expected


def test_levenshtein_is_symmetric() -> None:
    assert levenshtein("コメ", "コメノ") == levenshtein("コメノ", "コメ")


def test_accuracy_is_exact_match_ratio() -> None:
    assert accuracy(["コメ", "ベイ", "コメ"], ["コメ", "ベイ", "ベイ"]) == pytest.approx(2 / 3)


def test_accuracy_all_correct() -> None:
    assert accuracy(["コメ"], ["コメ"]) == 1.0


def test_accuracy_empty() -> None:
    assert accuracy([], []) == 0.0


def test_accuracy_any_accepts_multiple_gold_readings() -> None:
    """★benchmark は readings.natural に複数の正解を持つ（憧憬→ショウケイ/ドウケイ）。"""
    preds = ["ショウ", "セイ"]
    golds = [("ショウ", "ドウ"), ("ショウ", "ドウ")]

    assert accuracy_any(preds, golds) == pytest.approx(0.5)


def test_accuracy_any_matches_second_candidate() -> None:
    assert accuracy_any(["ドウ"], [("ショウ", "ドウ")]) == 1.0


def test_corpus_per_sums_over_corpus() -> None:
    """★文ごとに正規化して平均するのではなく、コーパス全体で合算すること。"""
    # 編集距離の合計 2、正解の総文字数 2 + 5 = 7
    assert corpus_per(["コメ", "アイウエオ"], ["ベイ", "アイウエオ"]) == pytest.approx(2 / 7)


def test_corpus_per_zero_when_perfect() -> None:
    assert corpus_per(["コメ", "ベイ"], ["コメ", "ベイ"]) == 0.0


def test_corpus_per_empty() -> None:
    assert corpus_per([], []) == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="件数"):
        accuracy(["コメ"], ["コメ", "ベイ"])


def test_eval_result_formats() -> None:
    out = EvalResult(accuracy=0.99, target_per=0.003, sentence_per=0.001, n=100).format()
    assert "99.00%" in out
    assert "100" in out
