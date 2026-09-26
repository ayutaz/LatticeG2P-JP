from lattice_g2p.data.consistency import (
    ConsistencyReport,
    check_label_consistency,
    drop_inconsistent,
)
from lattice_g2p.data.schema import TrainingExample


def _ex(surface: str, reading: str, text: str = "") -> TrainingExample:
    return TrainingExample(
        text=text or f"{surface}を見る",
        reading="ダミー",
        target_start=0,
        target_end=len(surface),
        target_surface=surface,
        target_reading=reading,
    )


def test_single_reading_is_consistent() -> None:
    report = check_label_consistency([_ex("米", "コメ"), _ex("米", "コメ", "米が実る")])

    assert report.conflicts == {}


def test_detects_minority_reading_as_conflict() -> None:
    """★同じ語に読みが複数あり、片方が極端に少ないなら混乱とみなす。

    「致仕 → チシ 1 件 / チジ 3 件」のようなケース。
    どちらも辞書にあるためフィルタでは落ちない。
    """
    rows = [
        _ex("致仕", "チシ", "致仕の時期"),
        _ex("致仕", "チジ", "致仕を考える"),
        _ex("致仕", "チジ", "致仕後の生活"),
        _ex("致仕", "チジ", "致仕を願い出る"),
    ]
    report = check_label_consistency(rows)

    assert "致仕" in report.conflicts
    assert report.conflicts["致仕"] == ["チシ"], "少数派だけを落とす"


def test_balanced_readings_are_kept() -> None:
    """★意図的な多音語は落とさないこと（部屋 → ヘヤ / ベヤ）。"""
    rows = [_ex("部屋", "ヘヤ", f"部屋{i}") for i in range(4)] + [
        _ex("部屋", "ベヤ", f"大部屋{i}") for i in range(4)
    ]
    report = check_label_consistency(rows, min_ratio=0.3)

    assert report.conflicts == {}


def test_many_balanced_readings_are_kept() -> None:
    """★★読みが多い多音語を落とさないこと。

    「八」は ハチ / ハッ / ヤ / ヤッ / ヨウ の 5 通りを持つ。
    各読みが均等に 4 件ずつでも、合計に対する割合は 20% にしかならない。
    「合計に対する割合」で判定すると、最も価値の高い多音語が全滅する。
    """
    readings = ["ハチ", "ハッ", "ヤ", "ヤッ", "ヨウ"]
    rows = [_ex("八", r, f"八{r}{i}") for r in readings for i in range(4)]
    report = check_label_consistency(rows, min_ratio=0.3)

    assert report.conflicts == {}, "均等に分布していれば矛盾ではない"


def test_orthographic_variants_are_not_conflicts() -> None:
    """長音の書き分けだけの違いは矛盾ではない（コウ ≡ コー）。"""
    rows = [_ex("亢", "コウ", f"亢{i}") for i in range(5)] + [_ex("亢", "コー", "亢進")]
    report = check_label_consistency(rows, min_ratio=0.3)

    assert report.conflicts == {}


def test_drop_inconsistent_removes_minority() -> None:
    rows = [
        _ex("致仕", "チシ", "致仕の時期"),
        _ex("致仕", "チジ", "致仕を考える"),
        _ex("致仕", "チジ", "致仕後の生活"),
        _ex("致仕", "チジ", "致仕を願い出る"),
    ]
    kept, report = drop_inconsistent(rows)

    assert len(kept) == 3
    assert all(r.target_reading == "チジ" for r in kept)
    assert report.dropped == 1


def test_drop_inconsistent_keeps_everything_when_clean() -> None:
    rows = [_ex("米", "コメ", f"米{i}") for i in range(3)]
    kept, report = drop_inconsistent(rows)

    assert len(kept) == 3
    assert report.dropped == 0


def test_report_formats() -> None:
    report = ConsistencyReport(total=10, dropped=2, conflicts={"致仕": ["チシ"]})
    out = report.format()

    assert "10" in out
    assert "致仕" in out


def test_min_count_guards_against_tiny_samples() -> None:
    """★出現が 1 回だけの語は判断材料が足りない。落とさないこと。"""
    rows = [_ex("稀語", "キゴ", "稀語を使う")]
    kept, report = drop_inconsistent(rows, min_count=2)

    assert len(kept) == 1
    assert report.dropped == 0
