"""評価指標. 定義は docs/evaluation.md §1 に従う.

★論文は PER (Phoneme Error Rate) と呼ぶが, 実態はカナ列に対する CER である.
"""

from collections.abc import Sequence
from dataclasses import dataclass


def levenshtein(a: str, b: str) -> int:
    """編集距離 (挿入・削除・置換のコストを 1 とする)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _check_lengths(preds: Sequence[object], golds: Sequence[object]) -> None:
    if len(preds) != len(golds):
        raise ValueError(f"件数が一致しません: preds={len(preds)}, golds={len(golds)}")


def accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    """完全一致の割合."""
    _check_lengths(preds, golds)
    if not golds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds, strict=True)) / len(golds)


def accuracy_any(preds: Sequence[str], golds: Sequence[Sequence[str]]) -> float:
    """正解候補のいずれかに一致すれば正解とする Accuracy.

    benchmark の ``readings.natural`` は複数ありうる.
    「憧憬」が「ショウケイ」とも「ドウケイ」とも読めるように, 文脈から
    一意に決まらない場合が存在するため (docs/evaluation.md §1-1).
    """
    _check_lengths(preds, golds)
    if not golds:
        return 0.0
    return sum(p in set(g) for p, g in zip(preds, golds, strict=True)) / len(golds)


def corpus_per(preds: Sequence[str], golds: Sequence[str]) -> float:
    """コーパス全体で合算した編集距離ベースの誤り率.

        sum(levenshtein) / sum(len(gold))

    文ごとに正規化して平均するのではない点に注意.
    """
    _check_lengths(preds, golds)
    total_dist = sum(levenshtein(p, g) for p, g in zip(preds, golds, strict=True))
    total_len = sum(len(g) for g in golds)
    if total_len == 0:
        return 0.0
    return total_dist / total_len


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    target_per: float
    sentence_per: float
    n: int

    def format(self) -> str:
        return (
            f"Accuracy      : {self.accuracy * 100:.2f}%\n"
            f"Target PER    : {self.target_per * 100:.2f}%\n"
            f"Sentence PER  : {self.sentence_per * 100:.2f}%\n"
            f"件数          : {self.n:,}"
        )
