import pytest
import torch

from lattice_g2p.crf import viterbi
from lattice_g2p.decode import path_reading, span_reading
from lattice_g2p.dictionary import Dictionary, Entry
from lattice_g2p.lattice import build_lattice


@pytest.fixture
def dic() -> Dictionary:
    return Dictionary.from_entries(
        [
            Entry("米", "コメ", "名詞", 0),
            Entry("米", "ベイ", "名詞", 1),
            Entry("米国", "ベイコク", "名詞", 2),
            Entry("国", "クニ", "名詞", 3),
            Entry("の", "ノ", "助詞", 4),
            Entry("。", "*", "補助記号-句点", 5),
        ]
    )


def _node_index(lat, start: int, reading: str) -> int:  # noqa: ANN001
    return next(i for i, n in enumerate(lat.nodes) if (n.start, n.reading) == (start, reading))


def test_path_reading_concatenates(dic: Dictionary) -> None:
    lat = build_lattice("米の", dic)
    path = [_node_index(lat, 0, "コメ"), _node_index(lat, 1, "ノ")]

    assert path_reading(lat, path) == "コメノ"


def test_path_reading_skips_empty_readings(dic: Dictionary) -> None:
    """句読点の空読みは読み列に寄与しないこと。"""
    lat = build_lattice("米。", dic)
    path = [_node_index(lat, 0, "コメ"), _node_index(lat, 1, "")]

    assert path_reading(lat, path) == "コメ"


def test_path_reading_empty_path(dic: Dictionary) -> None:
    assert path_reading(build_lattice("", dic), []) == ""


def test_span_reading_single_node(dic: Dictionary) -> None:
    lat = build_lattice("米の", dic)
    path = [_node_index(lat, 0, "コメ"), _node_index(lat, 1, "ノ")]

    assert span_reading(lat, path, 0, 1) == "コメ"


def test_span_reading_concatenates_split_nodes(dic: Dictionary) -> None:
    """★target span が複数ノードに分割されていても連結して返すこと。"""
    lat = build_lattice("米国", dic)
    path = [_node_index(lat, 0, "ベイ"), _node_index(lat, 1, "クニ")]

    assert span_reading(lat, path, 0, 2) == "ベイクニ"


def test_span_reading_includes_straddling_node(dic: Dictionary) -> None:
    """span をまたぐノードは丸ごと含める（規約）。"""
    lat = build_lattice("米国", dic)
    path = [_node_index(lat, 0, "ベイコク")]

    assert span_reading(lat, path, 1, 2) == "ベイコク"


def test_span_reading_excludes_outside_nodes(dic: Dictionary) -> None:
    lat = build_lattice("米の", dic)
    path = [_node_index(lat, 0, "コメ"), _node_index(lat, 1, "ノ")]

    assert span_reading(lat, path, 1, 2) == "ノ"


def test_viterbi_covers_the_whole_text(dic: Dictionary) -> None:
    lat = build_lattice("米国の", dic)
    torch.manual_seed(0)
    path = viterbi(lat, torch.randn(len(lat.nodes)))

    assert lat.nodes[path[0]].start == 0
    assert lat.nodes[path[-1]].end == len(lat.text)
    for a, b in zip(path, path[1:], strict=False):
        assert lat.nodes[a].end == lat.nodes[b].start
