import json
from pathlib import Path

import pytest

from lattice_g2p.benchmark_data import BenchmarkItem, load_joyo_benchmark, split_tagged

_ROW = {
    "key": "固_かためる_0",
    "text": "チームの結束を固める必要がある。",
    "tagged_text": "チームの結束を<固>める必要がある。",
    "yomi": "チームノケッソクヲカタメルヒツヨウガアル。",
    "tagged_yomi": "チームノケッソクヲ<カタ>メルヒツヨウガアル。",
    "reading_category": "kun_yomi",
    "readings": {"natural": ["カタ"], "marginal": []},
    "source": "original_fixed",
}


def _write(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "bench.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def test_split_tagged_returns_span_and_inner() -> None:
    plain, start, end = split_tagged("チームの結束を<固>める")

    assert plain == "チームの結束を固める"
    assert (start, end) == (7, 8)
    assert plain[start:end] == "固"


def test_split_tagged_at_head() -> None:
    plain, start, end = split_tagged("<固>める")

    assert plain == "固める"
    assert (start, end) == (0, 1)


def test_split_tagged_multi_character_target() -> None:
    plain, start, end = split_tagged("彼は<眼鏡>をかけた")

    assert plain == "彼は眼鏡をかけた"
    assert plain[start:end] == "眼鏡"


def test_split_tagged_rejects_missing_tag() -> None:
    with pytest.raises(ValueError, match="タグ"):
        split_tagged("タグのない文")


def test_split_tagged_rejects_multiple_tags() -> None:
    with pytest.raises(ValueError, match="タグ"):
        split_tagged("<固>めて<固>める")


def test_load_single_row(tmp_path: Path) -> None:
    items = load_joyo_benchmark(_write(tmp_path, _ROW))

    assert len(items) == 1
    item = items[0]
    assert isinstance(item, BenchmarkItem)
    assert item.key == "固_かためる_0"
    assert item.text == "チームの結束を固める必要がある。"
    assert item.reading == "チームノケッソクヲカタメルヒツヨウガアル。"
    assert item.reading_category == "kun_yomi"


def test_target_span_points_into_text(tmp_path: Path) -> None:
    item = load_joyo_benchmark(_write(tmp_path, _ROW))[0]

    assert item.text[item.target_start : item.target_end] == "固"
    assert item.target_surface == "固"


def test_target_reading_comes_from_tagged_yomi(tmp_path: Path) -> None:
    item = load_joyo_benchmark(_write(tmp_path, _ROW))[0]

    assert item.target_reading == "カタ"


def test_acceptable_readings_include_natural(tmp_path: Path) -> None:
    """★readings.natural は複数ありうる。どれも正解として扱えること。"""
    row = dict(_ROW)
    row["readings"] = {"natural": ["ショウ", "ドウ"], "marginal": ["セイ"]}
    item = load_joyo_benchmark(_write(tmp_path, row))[0]

    assert set(item.natural_readings) == {"ショウ", "ドウ"}
    assert set(item.marginal_readings) == {"セイ"}


def test_tagged_text_consistent_with_text(tmp_path: Path) -> None:
    """tagged_text からタグを外したものが text と一致しない行は不正。"""
    row = dict(_ROW)
    row["tagged_text"] = "まったく別の<固>文"

    with pytest.raises(ValueError, match="text"):
        load_joyo_benchmark(_write(tmp_path, row))


def test_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "bench.jsonl"
    path.write_text(
        json.dumps(_ROW, ensure_ascii=False) + "\n\n" + json.dumps(_ROW, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert len(load_joyo_benchmark(path)) == 2


# --- 正規化後のカナ位置 ---


def test_target_kana_span_on_normalized_reading(tmp_path: Path) -> None:
    """★正解読みの中で対象語の読みが占めるカナ位置を持つこと。

    経路が target span をまたぐノードを選んだ場合、ノードの読みをそのまま
    取り出すと「固める→カタメル」のように対象外の読みまで含んでしまう。
    benchmark 自身が tagged_yomi でカナ位置を示しているので、それを使う。
    """
    item = load_joyo_benchmark(_write(tmp_path, _ROW))[0]
    normalized = "チームノケッソクヲカタメルヒツヨウガアル"

    assert normalized[item.target_kana_start : item.target_kana_end] == "カタ"


def test_target_kana_span_accounts_for_removed_punctuation(tmp_path: Path) -> None:
    """★正規化で句読点が消えるぶん、位置がずれること。"""
    row = dict(_ROW)
    row["text"] = "ああ、結束を固める。"
    row["tagged_text"] = "ああ、結束を<固>める。"
    row["yomi"] = "アア、ケッソクヲカタメル。"
    row["tagged_yomi"] = "アア、ケッソクヲ<カタ>メル。"
    item = load_joyo_benchmark(_write(tmp_path, row))[0]

    from lattice_g2p.kana import normalize_kana

    normalized = normalize_kana(item.reading)
    assert normalized == "アアケッソクヲカタメル"
    assert normalized[item.target_kana_start : item.target_kana_end] == "カタ"
