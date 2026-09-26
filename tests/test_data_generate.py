from pathlib import Path

import pytest

from lattice_g2p.data.generate import generate_reading_groups, generate_sentences
from lattice_g2p.data.schema import ReadingGroup, read_jsonl
from lattice_g2p.data.select import PolyphonicWord


class FakeClient:
    """呼び出しを記録するだけのクライアント。API を叩かない。"""

    model = "fake-model"

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def call_tool(self, prompt: str, tool: dict) -> dict:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("想定より多く呼ばれました")
        return self._responses.pop(0)


def _word() -> PolyphonicWord:
    return PolyphonicWord(surface="米", readings=("コメ", "ベイ"), pos_list=("名詞",))


def test_generate_reading_groups_writes_jsonl(tmp_path: Path) -> None:
    client = FakeClient(
        [
            {
                "groups": [
                    {"reading": "コメ", "valid": True, "sense": "穀物", "pos": "名詞"},
                    {"reading": "ベイ", "valid": True, "sense": "アメリカ", "pos": "名詞"},
                ]
            }
        ]
    )
    out = tmp_path / "groups.jsonl"

    generate_reading_groups(client, [_word()], out)

    rows = list(read_jsonl(out))
    assert len(rows) == 2
    assert {r["reading"] for r in rows} == {"コメ", "ベイ"}
    assert all(r["surface"] == "米" for r in rows)


def test_generate_reading_groups_normalizes_readings(tmp_path: Path) -> None:
    client = FakeClient(
        [{"groups": [{"reading": "こめ", "valid": True, "sense": "穀物", "pos": "名詞"}]}]
    )
    out = tmp_path / "groups.jsonl"

    generate_reading_groups(client, [_word()], out)

    assert list(read_jsonl(out))[0]["reading"] == "コメ"


def test_generate_reading_groups_resumes(tmp_path: Path) -> None:
    """★既に処理済みの語はスキップすること（生成は高コスト）。"""
    out = tmp_path / "groups.jsonl"
    client = FakeClient(
        [{"groups": [{"reading": "コメ", "valid": True, "sense": "穀物", "pos": "名詞"}]}]
    )
    generate_reading_groups(client, [_word()], out)
    assert len(client.prompts) == 1

    client2 = FakeClient([])
    generate_reading_groups(client2, [_word()], out)

    assert client2.prompts == [], "2 回目は API を呼ばないこと"
    assert len(list(read_jsonl(out))) == 1, "重複して書き込まないこと"


def test_generate_sentences_writes_jsonl(tmp_path: Path) -> None:
    groups = [
        ReadingGroup(surface="米", reading="コメ", sense="穀物", pos="名詞", valid=True),
        ReadingGroup(surface="米", reading="ハ", sense="", pos="", valid=False, reason="語幹"),
    ]
    client = FakeClient(
        [
            {
                "sentences": [
                    {"text": "米を炊く", "reading": "コメヲタク", "target_start": 0},
                ]
            }
        ]
    )
    out = tmp_path / "sentences.jsonl"

    generate_sentences(client, groups, out, n_per_group=1)

    rows = list(read_jsonl(out))
    assert len(rows) == 1, "valid=False のグループは生成対象にしないこと"
    assert rows[0]["text"] == "米を炊く"
    assert rows[0]["target_surface"] == "米"
    assert rows[0]["target_reading"] == "コメ"
    assert rows[0]["target_start"] == 0
    assert rows[0]["target_end"] == 1
    assert rows[0]["model"] == "fake-model"


def test_generate_sentences_resumes(tmp_path: Path) -> None:
    groups = [ReadingGroup("米", "コメ", "穀物", "名詞", True)]
    out = tmp_path / "sentences.jsonl"
    client = FakeClient(
        [{"sentences": [{"text": "米を炊く", "reading": "コメヲタク", "target_start": 0}]}]
    )
    generate_sentences(client, groups, out, n_per_group=1)

    client2 = FakeClient([])
    generate_sentences(client2, groups, out, n_per_group=1)

    assert client2.prompts == []
    assert len(list(read_jsonl(out))) == 1


def test_generate_sentences_skips_malformed_rows(tmp_path: Path) -> None:
    """壊れた応答で全体を落とさないこと。"""
    groups = [ReadingGroup("米", "コメ", "穀物", "名詞", True)]
    client = FakeClient(
        [
            {
                "sentences": [
                    {"text": "米を炊く", "reading": "コメヲタク"},  # target_start 欠落
                    {"text": "米が実る", "reading": "コメガミノル", "target_start": 0},
                ]
            }
        ]
    )
    out = tmp_path / "sentences.jsonl"

    generate_sentences(client, groups, out, n_per_group=2)

    rows = list(read_jsonl(out))
    assert len(rows) == 1
    assert rows[0]["text"] == "米が実る"


def test_generate_sentences_records_prompt_version(tmp_path: Path) -> None:
    from lattice_g2p.data.prompts import PROMPT_VERSION

    groups = [ReadingGroup("米", "コメ", "穀物", "名詞", True)]
    client = FakeClient(
        [{"sentences": [{"text": "米を炊く", "reading": "コメヲタク", "target_start": 0}]}]
    )
    out = tmp_path / "sentences.jsonl"

    generate_sentences(client, groups, out, n_per_group=1)

    assert list(read_jsonl(out))[0]["prompt_version"] == PROMPT_VERSION


def test_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from lattice_g2p.data.client import ClaudeClient

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("lattice_g2p.data.client.load_dotenv", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeClient()
