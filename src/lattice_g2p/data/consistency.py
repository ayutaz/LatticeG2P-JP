"""学習データのラベル整合性チェック.

★フィルタを通過しても, 同じ語に矛盾する読みが付いていることがある.

    致仕 -> チシ  （1 件）
    致仕 -> チジ  （3 件）    ほぼ同じ文脈なのに違う読み

どちらも辞書にある読みなのでラティス整合チェックでは落ちない.
しかし「ほぼ同じ文脈で違う読みを教える」データは学習を妨げる.
生成モデルが「文脈で一意に決まらない読み」を除外しきれなかったときに起きる.

★意図的な多音語 (部屋 -> ヘヤ / ベヤ) と区別すること.
意図的なものは両方が相応の件数あり, 文脈が明確に違う.
混乱によるものは片方が極端に少ない.

判定は件数比で行う. 完璧ではないが, 目視で見つけた実際の誤りを機械的に
拾えるだけの精度はある.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from lattice_g2p.data.schema import TrainingExample
from lattice_g2p.kana import phonetic_key

DEFAULT_MIN_RATIO = 0.4
"""最頻の読みに対する件数比. これを下回る読みを少数派 (混乱) とみなす.

★「合計に対する割合」で判定してはいけない.
「八」は ハチ / ハッ / ヤ / ヤッ / ヨウ の 5 通りを持ち, 各読みが均等に
4 件ずつあっても合計に対しては 20% にしかならない. 合計比で判定すると
最も価値の高い多音語が全滅する (実測で Phase A の 11.5% が落ちた).

最頻の読みと比べれば, 均等に分布している多音語は残り,
片方だけ極端に少ない混乱だけが落ちる.
"""

DEFAULT_MIN_COUNT = 2
"""この件数を下回る語は判断材料が足りないので対象外."""


@dataclass
class ConsistencyReport:
    total: int = 0
    dropped: int = 0
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    """表記 -> 落とした読みの一覧."""

    def format(self) -> str:
        rate = self.dropped / self.total * 100 if self.total else 0.0
        lines = [
            f"整合性チェック  {self.total:,} 件中 {self.dropped:,} 件を除外 ({rate:.1f}%)",
        ]
        if self.conflicts:
            lines.append(f"矛盾のあった語: {len(self.conflicts)} 語")
            for surface, readings in list(self.conflicts.items())[:15]:
                lines.append(f"  {surface}: {readings} を除外")
        return "\n".join(lines)


def check_label_consistency(
    rows: list[TrainingExample],
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_count: int = DEFAULT_MIN_COUNT,
) -> ConsistencyReport:
    """同じ語に付いた読みの分布を見て, 少数派を矛盾として報告する."""
    by_surface: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        # 長音の書き分けは矛盾ではないので畳んでから数える
        by_surface[row.target_surface][phonetic_key(row.target_reading)] += 1

    conflicts: dict[str, list[str]] = {}
    for surface, counter in by_surface.items():
        if len(counter) < 2 or sum(counter.values()) < min_count:
            continue

        top = max(counter.values())
        minority_keys = {k for k, n in counter.items() if n < top * min_ratio}
        if not minority_keys:
            continue

        # 表示用に元の表記を拾い直す
        dropped_readings = sorted(
            {
                row.target_reading
                for row in rows
                if row.target_surface == surface
                and phonetic_key(row.target_reading) in minority_keys
            }
        )
        conflicts[surface] = dropped_readings

    return ConsistencyReport(total=len(rows), conflicts=conflicts)


def drop_inconsistent(
    rows: list[TrainingExample],
    min_ratio: float = DEFAULT_MIN_RATIO,
    min_count: int = DEFAULT_MIN_COUNT,
) -> tuple[list[TrainingExample], ConsistencyReport]:
    """矛盾する少数派の読みを持つ例を除外する."""
    report = check_label_consistency(rows, min_ratio, min_count)
    if not report.conflicts:
        return list(rows), report

    dropped_keys = {
        (surface, phonetic_key(reading))
        for surface, readings in report.conflicts.items()
        for reading in readings
    }

    kept = [
        row
        for row in rows
        if (row.target_surface, phonetic_key(row.target_reading)) not in dropped_keys
    ]
    report.dropped = len(rows) - len(kept)
    return kept, report
