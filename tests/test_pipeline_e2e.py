"""M2 パイプライン全体の結合テスト.

select -> generate(Fake) -> filter -> prepare -> CRF 損失 まで通し、
生成データが実際に学習できる形になっていることを確認する。
"""

from pathlib import Path

import torch

from lattice_g2p.crf import crf_loss_from_prepared, viterbi
from lattice_g2p.data.filter import filter_sentences
from lattice_g2p.data.generate import generate_reading_groups, generate_sentences
from lattice_g2p.data.prepare import prepare_all
from lattice_g2p.data.schema import GeneratedSentence, ReadingGroup, read_jsonl
from lattice_g2p.data.select import select_polyphonic
from lattice_g2p.decode import path_reading
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.scorer import UnigramScorer


class ScriptedClient:
    """決まった応答を返すクライアント。API を叩かない。"""

    model = "scripted"

    def __init__(self, groups: dict, sentences: dict) -> None:
        self._groups = groups
        self._sentences = sentences

    def call_tool(self, prompt: str, tool: dict) -> dict:
        if tool["name"] == "report_reading_groups":
            for surface, payload in self._groups.items():
                if f"「{surface}」" in prompt:
                    return payload
            return {"groups": []}
        for key, payload in self._sentences.items():
            if f"「{key[0]}」を「{key[1]}」" in prompt:
                return payload
        return {"sentences": []}


def _dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞-普通名詞-一般", 0, cost=1000),
            Entry("米", "ベイ", "名詞-普通名詞-一般", 0, cost=1100),
            Entry("を", "ヲ", "助詞-格助詞", 0, cost=-1000),
            Entry("炊く", "タク", "動詞-一般", 0, cost=2000),
            Entry("国", "コク", "名詞-普通名詞-一般", 0, cost=1500),
            Entry("の", "ノ", "助詞-格助詞", 0, cost=-900),
            Entry("政策", "セイサク", "名詞-普通名詞-一般", 0, cost=1500),
            Entry("。", "*", "補助記号-句点", 0, cost=-3000),
        ]
    )


def test_pipeline_produces_trainable_data(tmp_path: Path) -> None:
    dic = _dic()

    # 1. 多音語の抽出
    words = select_polyphonic(dic)
    assert [w.surface for w in words] == ["米"]

    # 2. 読みの検証
    client = ScriptedClient(
        groups={
            "米": {
                "groups": [
                    {"reading": "コメ", "valid": True, "sense": "穀物", "pos": "名詞"},
                    {"reading": "ベイ", "valid": True, "sense": "アメリカ", "pos": "名詞"},
                ]
            }
        },
        sentences={
            ("米", "コメ"): {
                "sentences": [
                    {"text": "米を炊く。", "reading": "コメヲタク", "target_start": 0}
                ]
            },
            ("米", "ベイ"): {
                "sentences": [
                    {"text": "米国の政策", "reading": "ベイコクノセイサク", "target_start": 0}
                ]
            },
        },
    )
    groups_path = tmp_path / "groups.jsonl"
    generate_reading_groups(client, words, groups_path, progress_every=0)
    groups = [ReadingGroup(**r) for r in read_jsonl(groups_path)]
    assert len(groups) == 2

    # 3. 文生成
    sentences_path = tmp_path / "sentences.jsonl"
    generate_sentences(client, groups, sentences_path, n_per_group=1, progress_every=0)
    rows = [GeneratedSentence(**r) for r in read_jsonl(sentences_path)]
    assert len(rows) == 2

    # 4. フィルタ
    kept, stats = filter_sentences(rows, dic, min_len=4)
    assert stats.lattice_ok == 2, "ラティス整合をすべて通ること"
    assert len(kept) == 2

    # 5. 前処理
    prepared, dropped = prepare_all(kept, dic)
    assert dropped == 0
    assert len(prepared) == 2

    # 6. ★学習できる形になっているか（CRF 損失が有限で非負）
    scorer = UnigramScorer()
    for p in prepared:
        scores = scorer.score(p.text, p.nodes)
        loss = crf_loss_from_prepared(p, scores)
        assert torch.isfinite(loss), f"損失が発散: {p.text}"
        assert float(loss) >= -1e-6, f"損失が負: {float(loss)}"

    # 7. 正解読みがラティス上で再現できるか
    for p in prepared:
        lat = p.lattice()
        gold_scores = torch.zeros(len(p.nodes))
        for idx in p.gold_node_indices:
            gold_scores[idx] = 100.0
        assert path_reading(lat, viterbi(lat, gold_scores)) == p.gold_reading


def test_pipeline_drops_inconsistent_generation(tmp_path: Path) -> None:
    """★辞書で生成できない読みが混ざっても、学習データに漏れないこと。"""
    dic = _dic()
    client = ScriptedClient(
        groups={
            "米": {
                "groups": [
                    {"reading": "コメ", "valid": True, "sense": "穀物", "pos": "名詞"}
                ]
            }
        },
        sentences={
            ("米", "コメ"): {
                "sentences": [
                    {"text": "米を炊く。", "reading": "コメヲタク", "target_start": 0},
                    {"text": "米を炊く。", "reading": "イリガビタク", "target_start": 0},
                ]
            }
        },
    )
    groups_path = tmp_path / "g.jsonl"
    sentences_path = tmp_path / "s.jsonl"
    generate_reading_groups(client, select_polyphonic(dic), groups_path, progress_every=0)
    groups = [ReadingGroup(**r) for r in read_jsonl(groups_path)]
    generate_sentences(client, groups, sentences_path, n_per_group=2, progress_every=0)

    rows = [GeneratedSentence(**r) for r in read_jsonl(sentences_path)]
    kept, stats = filter_sentences(rows, dic, min_len=4)

    assert stats.format_ok == 2
    assert stats.lattice_ok == 1, "辞書で作れない読みはラティス整合で落ちること"
    assert len(kept) == 1

    _, dropped = prepare_all(kept, dic)
    assert dropped == 0, "フィルタを通った例は必ず前処理できること"
