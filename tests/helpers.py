"""テスト専用の素朴実装.

★本番コードから絶対に import しないこと. 経路数が指数的に増える.
CRF の forward が正しいことを証明するためだけに使う.
"""

from lattice_g2p.lattice import Lattice


def enumerate_paths(lattice: Lattice) -> list[list[int]]:
    """0 から len(text) への全経路を列挙する (ノード index のリスト)."""
    goal = len(lattice.text)
    results: list[list[int]] = []
    acc: list[int] = []

    def rec(pos: int) -> None:
        if pos == goal:
            results.append(list(acc))
            return
        for node_idx in lattice.nodes_starting_at[pos]:
            acc.append(node_idx)
            rec(lattice.nodes[node_idx].end)
            acc.pop()

    rec(0)
    return results


def _reading_of(lattice: Lattice, path: list[int]) -> str:
    """★lattice_g2p.decode.path_reading とは別物.

    本番実装に依存せずテストが独立して正しさを判定できるよう, あえて別実装にしている.
    名前も衝突させない.
    """
    return "".join(lattice.nodes[i].reading for i in path)


def gold_paths(lattice: Lattice, gold_reading: str) -> list[list[int]]:
    """正解読みを生成する経路だけを返す."""
    return [p for p in enumerate_paths(lattice) if _reading_of(lattice, p) == gold_reading]


def paths_satisfying_spans(
    lattice: Lattice, spans: list[tuple[int, int, str]]
) -> list[list[int]]:
    """指定した span だけ読みが一致する経路を返す (素朴な全列挙).

    ★仕様の定義そのもの. 積グラフ実装の正しさはこれと突き合わせて確認する.

    span の境界にノードの境界が来ることを要求する (分離できない経路は認めない).
    """
    results: list[list[int]] = []
    for path in enumerate_paths(lattice):
        boundaries = {0}
        pos = 0
        for node_idx in path:
            pos = lattice.nodes[node_idx].end
            boundaries.add(pos)

        ok = True
        for start, end, reading in spans:
            if start not in boundaries or end not in boundaries:
                ok = False
                break
            inside = "".join(
                lattice.nodes[i].reading
                for i in path
                if lattice.nodes[i].start >= start and lattice.nodes[i].end <= end
            )
            if inside != reading:
                ok = False
                break
        if ok:
            results.append(path)
    return results
