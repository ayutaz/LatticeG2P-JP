"""CPU レイテンシの計測.

★必ずローカルの固定環境で実行する (docs/pitfalls.md R-13).
vast.ai のインスタンスは CPU を他テナントと共有し, 型番も vCPU 割り当ても
不定なので, そこで測った値には再現性がない. それらしい数字が出るぶん
気づきにくい.

★平均値だけを報告しない. TTS の体感に効くのは p95.
★内訳を必ず取る. scoring が支配的でないならバッチ化の失敗を疑う (R-12).
"""

import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

SECTIONS = ("lattice", "context", "scoring", "viterbi")


@dataclass
class BenchmarkResult:
    label: str
    n_chars: int
    n_trials: int
    n_nodes_mean: float
    p50: float
    p95: float
    breakdown_p50: dict[str, float] = field(default_factory=dict)
    peak_memory_mb: float = 0.0

    def format(self) -> str:
        lines = [
            f"文字数        : {self.n_chars}",
            f"平均ノード数   : {self.n_nodes_mean:.1f}",
            f"p50           : {self.p50:.2f} ms",
            f"p95           : {self.p95:.2f} ms",
            f"peak memory   : {self.peak_memory_mb:.1f} MB",
        ]
        if self.breakdown_p50:
            lines.append("内訳 (p50):")
            for key in SECTIONS:
                value = self.breakdown_p50.get(key, 0.0)
                share = value / self.p50 * 100 if self.p50 else 0.0
                lines.append(f"  {key:<10}{value:6.2f} ms  ({share:4.1f}%)")
        return "\n".join(lines)


def environment_info(model_dir: Path | None = None) -> str:
    lines = [
        f"platform     : {platform.platform()}",
        f"processor    : {platform.processor() or platform.machine()}",
        f"python       : {platform.python_version()}",
    ]
    if model_dir is not None:
        size = sum(f.stat().st_size for f in Path(model_dir).rglob("*") if f.is_file()) / 1e6
        lines.append(f"model size   : {size:.1f} MB")
    return "\n".join(lines)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(len(sorted_values) * q))
    return sorted_values[index]


def measure(
    fn: Callable[[str], tuple[object, dict[str, float] | None]],
    texts: list[str],
    label: str,
    warmup: int = 10,
    trials: int = 100,
    n_nodes_mean: float = 0.0,
) -> BenchmarkResult:
    """fn(text) -> (結果, 内訳 or None) を繰り返し呼んで分布を取る.

    ★tracemalloc は計測ループの外で回す. 有効にしたまま計ると
    すべての alloc にフックが入り, レイテンシが歪む.
    """
    for i in range(warmup):
        fn(texts[i % len(texts)])

    totals: list[float] = []
    parts: dict[str, list[float]] = {k: [] for k in SECTIONS}
    for i in range(trials):
        text = texts[i % len(texts)]
        t0 = time.perf_counter()
        _, timing = fn(text)
        totals.append((time.perf_counter() - t0) * 1000)
        if timing:
            for key in parts:
                parts[key].append(timing[key])

    tracemalloc.start()
    fn(texts[0])
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    totals.sort()
    return BenchmarkResult(
        label=label,
        n_chars=len(texts[0]),
        n_trials=trials,
        n_nodes_mean=n_nodes_mean,
        p50=statistics.median(totals),
        p95=_percentile(totals, 0.95),
        breakdown_p50={k: statistics.median(v) for k, v in parts.items() if v},
        peak_memory_mb=peak,
    )


def run_benchmark(g2p, texts: list[str], **kw) -> BenchmarkResult:  # noqa: ANN001, ANN003
    """OnnxG2P のレイテンシを内訳つきで測る."""
    from lattice_g2p.lattice import build_lattice

    nodes = statistics.mean(len(build_lattice(t, g2p.dic).nodes) for t in texts)
    return measure(
        g2p.g2p_with_timing, texts, label="OnnxG2P", n_nodes_mean=nodes, **kw
    )
