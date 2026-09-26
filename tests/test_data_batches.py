from pathlib import Path

from lattice_g2p.data.batches import make_batches, merge_batch_outputs, parse_batch_output
from lattice_g2p.data.schema import read_jsonl
from lattice_g2p.data.select import PolyphonicWord


def _words(n: int) -> list[PolyphonicWord]:
    return [
        PolyphonicWord(surface=f"語{i}", readings=("ア", "イ"), pos_list=("名詞",))
        for i in range(n)
    ]


def test_make_batches_splits_evenly(tmp_path: Path) -> None:
    paths = make_batches(_words(10), tmp_path, batch_size=4)

    assert len(paths) == 3
    assert [len(list(read_jsonl(p))) for p in paths] == [4, 4, 2]


def test_make_batches_writes_word_fields(tmp_path: Path) -> None:
    paths = make_batches(_words(1), tmp_path, batch_size=4)
    row = list(read_jsonl(paths[0]))[0]

    assert row["surface"] == "語0"
    assert row["readings"] == ["ア", "イ"]
    assert row["pos_list"] == ["名詞"]


def test_make_batches_is_deterministic(tmp_path: Path) -> None:
    a = make_batches(_words(6), tmp_path / "a", batch_size=2)
    b = make_batches(_words(6), tmp_path / "b", batch_size=2)

    assert [p.name for p in a] == [p.name for p in b]
    assert list(read_jsonl(a[0])) == list(read_jsonl(b[0]))


def test_parse_batch_output_reads_generated_sentences(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"text": "米を炊く", "reading": "コメヲタク", "target_start": 0,'
        ' "target_surface": "米", "target_reading": "コメ"}\n',
        encoding="utf-8",
    )

    rows = parse_batch_output(path, model="subagent")

    assert len(rows) == 1
    assert rows[0].text == "米を炊く"
    assert rows[0].target_end == 1
    assert rows[0].model == "subagent"


def test_parse_batch_output_skips_malformed(tmp_path: Path) -> None:
    """★サブエージェントの出力は壊れうる。1 行の欠損で全体を落とさないこと。"""
    path = tmp_path / "out.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"text": "米を炊く", "reading": "コメヲタク"}',  # target_start 欠落
                '{"text": "米が実る", "reading": "コメガミノル", "target_start": 0,'
                ' "target_surface": "米", "target_reading": "コメ"}',
                "壊れた行",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = parse_batch_output(path, model="subagent")

    assert [r.text for r in rows] == ["米が実る"]


def test_parse_batch_output_recomputes_target_start(tmp_path: Path) -> None:
    """★target_start がずれていても、表記から復元できるなら直すこと。"""
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"text": "今年の米は高い", "reading": "コトシノコメワタカイ", "target_start": 0,'
        ' "target_surface": "米", "target_reading": "コメ"}\n',
        encoding="utf-8",
    )

    rows = parse_batch_output(path, model="subagent")

    assert rows[0].target_start == 3
    assert rows[0].target_end == 4


def test_parse_batch_output_drops_when_surface_absent(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"text": "麦が実る", "reading": "ムギガミノル", "target_start": 0,'
        ' "target_surface": "米", "target_reading": "コメ"}\n',
        encoding="utf-8",
    )

    assert parse_batch_output(path, model="subagent") == []


def test_merge_batch_outputs(tmp_path: Path) -> None:
    for i in range(2):
        (tmp_path / f"batch_{i:03d}.out.jsonl").write_text(
            f'{{"text": "米を炊く{i}", "reading": "コメヲタク", "target_start": 0,'
            ' "target_surface": "米", "target_reading": "コメ"}\n',
            encoding="utf-8",
        )

    rows = merge_batch_outputs(tmp_path, model="subagent")

    assert len(rows) == 2
    assert {r.text for r in rows} == {"米を炊く0", "米を炊く1"}
