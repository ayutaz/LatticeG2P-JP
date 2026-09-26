"""dev のラベルを機械的に監査する.

★評価セットも実装物である. モデルと同じように疑う (docs/pitfalls.md R-19).

M3b で学習データを 4 倍にしても対象語精度が 68.02% -> 68.97% しか動かなかった.
「モデルが悪い」以外に「dev のラベルが悪い」可能性がある. 実際, 誤り例の目視で
疑わしいラベルが見つかっている.

    法師 : dev は ボウシ, モデルは ホウシ     ホウシ が標準的
    大麻 : dev は オオヌサ, モデルは タイマ   文脈上どちらも成立

★419 文すべてを目視するのは高い. まず機械で候補を絞る.

辞書のコストは頻度の代理指標になっている. ラベルがその表記の読みとして
低頻度なら, 疑わしい候補として挙げる.

★ここで挙がったものが「誤り」だとは限らない. dev は多音語のデータを狙って
作っているので, 低頻度の読みが正しい例も当然ある. あくまで目視の優先順位である.
"""

from dataclasses import dataclass, field
from pathlib import Path

from lattice_g2p.data.schema import read_jsonl
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import phonetic_key


def _fold(reading: str) -> str:
    """監査用の比較キー.

    ★四つ仮名 (ヅ/ズ・ヂ/ジ) は phonetic_key 側に移した.
    当初はここだけで畳んでいたが (「dev で 0 件なので精度に効かない」),
    benchmark では 33 件あり OpenJTalk との差の 5.1% を占めていた.
    二重定義を残さないこと (docs/architecture.md §9 不変条件 2).
    """
    return phonetic_key(reading)


DEFAULT_MIN_RANK_RATIO = 0.5
"""読みの順位 / 総数 がこれを超える例を候補にする.

1.0 は「その表記の最も低頻度な読み」. 値を上げるほど厳しく (候補が減る).
"""


@dataclass(frozen=True)
class AuditCandidate:
    """目視判定にかける候補 1 件."""

    text: str
    target_surface: str
    target_reading: str

    rank: tuple[int, int]
    """(順位, 読みの総数). 1 が最頻. ★0 は「辞書に無い読み」で最も疑わしい."""

    alternatives: list[str] = field(default_factory=list)
    """その表記の他の読み (コストの小さい順)."""


def reading_rank(dic: Dictionary, surface: str, reading: str) -> tuple[int, int]:
    """その読みが表記の中で何番目に頻度が高いか.

    ★長音の書き分け (コウ / コー) と四つ仮名 (カンヅメ / カンズメ) は
    同じ読みとして数える. 畳まないと異表記が別の読みに見え, 順位が実態とずれる
    (docs/pitfalls.md R-16).

    Returns:
        (順位, 読みの総数). 辞書に無い読みなら (0, 総数).
    """
    keys: list[str] = []
    for entry in sorted(dic.entries_of(surface), key=lambda e: e.cost):
        key = _fold(entry.reading)
        if key not in keys:
            keys.append(key)

    want = _fold(reading)
    return (keys.index(want) + 1 if want in keys else 0, len(keys))


def flag_suspicious(
    rows: list,  # noqa: ANN001  TrainingExample 互換なら何でもよい
    dic: Dictionary,
    min_rank_ratio: float = DEFAULT_MIN_RANK_RATIO,
) -> list[AuditCandidate]:
    """ラベルが低頻度の読みである例を候補に挙げる.

    ★読みが 1 つしかない表記は候補にしない. 疑いようがない.
    """
    out: list[AuditCandidate] = []
    for row in rows:
        rank, total = reading_rank(dic, row.target_surface, row.target_reading)
        if total < 2:
            continue
        # rank == 0 は辞書に無い読み. 最も疑わしいので必ず挙げる
        if rank and rank / total <= min_rank_ratio:
            continue
        out.append(
            AuditCandidate(
                text=row.text,
                target_surface=row.target_surface,
                target_reading=row.target_reading,
                rank=(rank, total),
                alternatives=_readings_by_cost(dic, row.target_surface),
            )
        )
    return out


def _readings_by_cost(dic: Dictionary, surface: str, limit: int = 6) -> list[str]:
    seen: list[str] = []
    for entry in sorted(dic.entries_of(surface), key=lambda e: e.cost):
        if entry.reading not in seen:
            seen.append(entry.reading)
        if len(seen) >= limit:
            break
    return seen


def load_accepts(path: Path) -> dict[tuple[str, str], list[str]]:
    """監査結果から「正解として認める読み」の対応表を作る.

        (表記, dev のラベル) -> 認める読みの一覧

    ★label_ok は入れない. dev のラベルのままで正しいので対応表は要らない.
    ★label_wrong は置き換える (dev のラベルを認めない).
    ★both_ok は dev のラベルと予測の両方を認める.

    ★同じ (表記, ラベル) が複数行に出ることがある. 監査は
    (表記, ラベル, 予測) 単位で分けたので, 予測が違えば別の行になる.
    評価側は予測を知らないので和を取る. わずかに甘くなる近似である.

    ★この対応表は、監査したモデルが実際に外した例だけを覆っている.
    別のモデルは別の例で外すので、そこに評価側の問題が残っている可能性がある.
    したがって補正後の数値は「評価側の問題を差し引いた下限」であって、
    上限ではない.
    """
    accepts: dict[tuple[str, str], list[str]] = {}
    for row in read_jsonl(path):
        if row.get("verdict") not in ("both_ok", "label_wrong"):
            continue
        key = (row["target_surface"], row["dev_label"])
        current = accepts.setdefault(key, [])
        for reading in row.get("accept", []):
            if reading not in current:
                current.append(reading)
    return accepts


def matches_accepted(
    predicted_span: str, target_reading: str, accepted: list[str] | None
) -> bool:
    """予測した span の読みが正解と認められるか.

    ★span を分離できない場合があるので部分文字列で見る (docs/pitfalls.md R-15).
    """
    if not predicted_span:
        return False
    wants = accepted if accepted else [target_reading]
    key = phonetic_key(predicted_span)
    return any(phonetic_key(w) in key for w in wants)
