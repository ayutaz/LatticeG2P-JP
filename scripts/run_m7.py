"""M7 Task 2: UniDic の連接行列そのものを差し込んで測る（学習しない）.

    uv run python scripts/run_m7.py --runs rep-size-m-42 rep-size-m-1337 rep-size-m-2024

構成（docs/results.md §1）:

    U0   UnigramScorer（node_penalty = 0）                      行列なし
    U    U0 + 連接行列（★重み 1.0 固定。選ばない）               MeCab 相当
    N0   ニューラル（INT8 ONNX・seed 3 本）                       行列なし
    N    N0 + 連接行列（重み = 各 checkpoint の dict_prior_weight）
    N    dev で選んだ重み（{0.05, 0.1, 0.2, 0.4}）               ★benchmark で選ばない
    感度分析: 補完・未知語ノードの連接を 0 / 行列の中央値にしたもの

★N0 と N は、同じ checkpoint のスコアに行列を足すかどうかだけが違う（対応のある比較）。
学習をしないので、この差にばらつきは無い。
★N0 は M6 で INT8 ONNX（models/rep-size-m-*-int8）を compare_systems.py で測った。
同じ物で測り直し、**既知の値が再現することを指標コードの検証にする**
（参考 Unigram 93.59 / N0 94.17・94.34・94.38 / OpenJTalk 96.85）。
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lattice_g2p.analysis import ErrorCategory, classify_error
from lattice_g2p.baselines import (
    find_unexpected_chars,
    openjtalk_readings,
    span_reading_from_morphemes,
)
from lattice_g2p.benchmark_data import BenchmarkItem, load_joyo_benchmark
from lattice_g2p.connection import ConnectionMatrix, unk_ids
from lattice_g2p.crf import viterbi_numpy
from lattice_g2p.data.audit import load_accepts, matches_accepted
from lattice_g2p.decode import path_reading, span_reading
from lattice_g2p.dictionary import Dictionary
from lattice_g2p.kana import normalize_kana, phonetic_key
from lattice_g2p.lattice import Lattice, Node, build_lattice
from lattice_g2p.metrics import corpus_per
from lattice_g2p.official_eval import evaluate_many
from lattice_g2p.runtime import OnnxScorer
from lattice_g2p.scorer import UnigramScorer
from lattice_g2p.scorer.matrix import MatrixTransitionScorer

sys.stdout.reconfigure(line_buffering=True)

HARD_SET_CATEGORIES = (
    "proper_noun_person",
    "proper_noun_place",
    "proper_noun_org",
    "numeral",
    "homograph",
)
SWEEP = (0.05, 0.1, 0.2, 0.4)
"""N の感度分析で dev に振る重み（計画 Task 2 Step 5）."""

OJ_NAME = "OpenJTalk"
REF_NAME = "参考: Unigram（M1 既定 node_penalty=30）"


# --- 構成 ---


@dataclass(frozen=True)
class Config:
    name: str
    base: str
    """スコアの出どころ. "U0" / "Uref" / run 名."""
    weight: float | None
    """連接行列の重み. None なら行列を足さない."""
    policy: str = "unk"
    """補完・未知語ノードの連接. "unk"（主案）/ "zero" / "median"（感度分析）."""


class _Fixed:
    """計算済みのスコアを返す. 同じ文を重みだけ変えて何度も解くため."""

    def __init__(self, scores: torch.Tensor) -> None:
        self.scores = scores

    def score(self, text: str, nodes: list[Node]) -> torch.Tensor:
        del text, nodes
        return self.scores


@dataclass
class Matrix:
    cm: ConnectionMatrix
    unk: dict[str, tuple[int, int]]
    median: int

    def solve(self, lat: Lattice, base_scores: torch.Tensor, cfg: Config) -> list[int]:
        if cfg.weight is None:
            return viterbi_numpy(lat, base_scores.numpy())
        scorer = MatrixTransitionScorer(
            _Fixed(base_scores),
            self.cm,
            cfg.weight,
            unk=self.unk if cfg.policy == "unk" else None,
            unresolved_cost={"unk": None, "zero": 0, "median": self.median}[cfg.policy],
        )
        return viterbi_numpy(
            lat,
            scorer.score(lat.text, lat.nodes).numpy(),
            scorer.transitions(lat.nodes).numpy(),
        )


def median_cost(cm: ConnectionMatrix) -> int:
    """行列全体の連接コストの中央値. 補完ノードの感度分析（定数）に使う."""
    flat = cm.table.reshape(-1)
    counts = np.zeros(1 << 16, dtype=np.int64)
    chunk = 1 << 24
    for i in range(0, flat.size, chunk):
        counts += np.bincount(
            np.asarray(flat[i : i + chunk], dtype=np.int64) + (1 << 15), minlength=1 << 16
        )
    return int(np.searchsorted(np.cumsum(counts), (flat.size + 1) // 2)) - (1 << 15)


# --- 評価の規約（★compare_systems.py / run_hard_set.py と同じにすること） ---


def hit(span: str, item: BenchmarkItem) -> bool:
    if not span:
        return False
    accept = {phonetic_key(normalize_kana(r)) for r in item.acceptable_readings()}
    return any(a in phonetic_key(span) for a in accept)


def bench_metrics(preds: list[tuple[str, str]], items: list[BenchmarkItem]) -> dict[str, float]:
    sent = tgt = 0
    ps: list[str] = []
    gs: list[str] = []
    for (sentence, span), it in zip(preds, items, strict=True):
        gold = phonetic_key(normalize_kana(it.reading))
        ps.append(phonetic_key(sentence))
        gs.append(gold)
        sent += ps[-1] == gold
        tgt += hit(span, it)
    n = len(items)
    return {
        "文一致": sent / n * 100,
        "対象語": tgt / n * 100,
        "Sentence PER": corpus_per(ps, gs) * 100,
    }


def openjtalk_preds(items: list[BenchmarkItem]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    unexpected: set[str] = set()
    for it in items:
        morphemes = openjtalk_readings(it.text)
        sentence = "".join(r for _, r in morphemes)
        unexpected |= find_unexpected_chars(sentence)
        out.append(
            (sentence, span_reading_from_morphemes(morphemes, it.target_start, it.target_end))
        )
    if unexpected:
        # ★ここが空でないなら計測が歪んでいる（R-22）。黙って続けない
        raise SystemExit(f"★OpenJTalk の出力に未知の文字が残っている: {sorted(unexpected)}")
    return out


# --- 実行 ---


def run_items(
    items: list[BenchmarkItem],
    dic: Dictionary,
    bases: dict[str, object],
    configs: list[Config],
    matrix: Matrix,
    label: str,
) -> dict[str, list[tuple[str, str]]]:
    """各構成の (文全体の読み, 対象 span の読み). ベースのスコアは 1 文につき 1 回だけ計算する."""
    preds: dict[str, list[tuple[str, str]]] = {c.name: [] for c in configs}
    needed = sorted({c.base for c in configs})
    t0 = time.perf_counter()
    for k, it in enumerate(items):
        lat = build_lattice(it.text, dic)
        with torch.no_grad():
            scores = {b: bases[b].score(lat.text, lat.nodes) for b in needed}
        for cfg in configs:
            path = matrix.solve(lat, scores[cfg.base], cfg)
            preds[cfg.name].append(
                (
                    path_reading(lat, path),
                    span_reading(lat, path, it.target_start, it.target_end),
                )
            )
        if (k + 1) % 2000 == 0:
            print(f"  {label}: {k + 1:,} / {len(items):,}（{time.perf_counter() - t0:.0f} 秒）")
    return preds


def run_dev(
    rows: list[dict],
    dic: Dictionary,
    bases: dict[str, object],
    runs: list[str],
    weights: dict[str, list[float | None]],
    matrix: Matrix,
    accepts: dict[tuple[str, str], list[str]],
) -> dict[str, dict[str, dict[str, float]]]:
    """run -> 重み -> {対象語（監査後）, 文一致}. ★重みは dev で選ぶ."""
    hits = {r: {str(w): [0, 0] for w in weights[r]} for r in runs}
    for row in rows:
        lat = build_lattice(row["text"], dic)
        gold = normalize_kana(row["reading"])
        accepted = accepts.get((row["target_surface"], row["target_reading"]))
        for r in runs:
            with torch.no_grad():
                scores = bases[r].score(lat.text, lat.nodes)
            for w in weights[r]:
                path = matrix.solve(lat, scores, Config("dev", r, w))
                span = span_reading(lat, path, row["target_start"], row["target_end"])
                h = hits[r][str(w)]
                h[0] += matches_accepted(span, row["target_reading"], accepted)
                h[1] += path_reading(lat, path) == gold
    n = len(rows)
    return {
        r: {w: {"対象語": a / n * 100, "文一致": s / n * 100} for w, (a, s) in by_w.items()}
        for r, by_w in hits.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="checkpoints/ と models/ の run 名")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--unidic", type=Path, default=Path("data/unidic-src/unidic-cwj-3.1.1-full"))
    ap.add_argument("--dict-cache", type=Path, default=Path("data/dic_cache.pkl"))
    ap.add_argument(
        "--benchmark", type=Path, default=Path("data/benchmark/common_kanji_source.jsonl")
    )
    ap.add_argument("--hard-set", type=Path, default=Path("data/hard_set"))
    ap.add_argument(
        "--accepts",
        type=Path,
        default=Path("data/dev_audit_r21.jsonl"),
        help="dev_r21 の監査結果（★dev は checkpoint の設定から読む）",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/m7"))
    ap.add_argument("--limit", type=int, default=None, help="動作確認用")
    ap.add_argument(
        "--no-official",
        action="store_true",
        help="公式評価ツール（jkyb-eval）で測らない（★論文と比べるときは必要。R-28）",
    )
    args = ap.parse_args()
    runs: list[str] = args.runs

    dic = Dictionary.load(args.dict_cache)
    cm = ConnectionMatrix.load(args.unidic / "matrix.bin")
    matrix = Matrix(cm, unk_ids(args.unidic / "unk.def"), median_cost(cm))
    print(f"連接行列 {cm.lsize:,} × {cm.rsize:,} / 中央値 {matrix.median:,}")
    print(f"unk: KANJI {matrix.unk['KANJI']} / HIRAGANA {matrix.unk['HIRAGANA']}\n")

    # --- 重み: N は各 checkpoint が学習した dict_prior_weight（★benchmark で選ばない） ---
    prior: dict[str, float] = {}
    dev_paths: set[str] = set()
    for r in runs:
        ckpt = torch.load(
            args.checkpoints_dir / r / "best.pt", map_location="cpu", weights_only=False
        )
        prior[r] = float(ckpt["model"]["dict_prior_weight"])
        dev_paths.add(str(ckpt["config"]["dev_path"]))
        if ckpt["config"].get("transition_dim", 0):
            raise SystemExit(f"{r} は遷移項つきで学習されている。M7 の N0 は遷移なしの前提")
    print("dict_prior_weight:", {r: round(w, 4) for r, w in prior.items()})
    # ★dev は学習時に使ったものを使う. data/dev.jsonl は M3d 以前の 419 文で母集団が違う
    if len(dev_paths) != 1:
        raise SystemExit(f"run ごとに dev が違う: {sorted(dev_paths)}")
    dev_path = Path(dev_paths.pop())

    def make_bases(fp32: bool = True) -> dict[str, object]:
        """★段ごとに作り直す. INT8 の読みキャッシュは評価の履歴に依存する.

        動的量子化は活性の尺度をバッチの min / max で決めるので、同じ読みでも
        「どの文と同じバッチで最初に計算されたか」でベクトルが変わる
        （同じ文のノードスコアが最大 0.12 ずれる。FP32 はずれない）。
        作り直せば compare_systems.py / run_hard_set.py と同じ順序で計算される.
        """
        bases: dict[str, object] = {
            "U0": UnigramScorer(node_penalty=0.0),
            "Uref": UnigramScorer(),  # M1 の既定値（node_penalty = 30）
        }
        for r in runs:
            bases[r] = OnnxScorer(args.models_dir / f"{r}-int8", dic)
            if fp32:
                bases[f"{r}-fp32"] = OnnxScorer(args.models_dir / f"{r}-fp32", dic)
        return bases

    # --- dev で N の重みを選ぶ（INT8） ---
    rows = [json.loads(line) for line in dev_path.open(encoding="utf-8")]
    accepts = load_accepts(args.accepts) if args.accepts.exists() else {}
    dev_bases = make_bases(fp32=False)
    dev_weights = {r: [None, prior[r], *SWEEP] for r in runs}
    dev = run_dev(rows, dic, dev_bases, runs, dev_weights, matrix, accepts)
    print(f"\n=== dev {len(rows)} 文（{dev_path} / 監査 {args.accepts}）===")
    print("  重み                  対象語（監査後） / 文一致  ※seed ごと")
    for w in ["None", "prior", *map(str, SWEEP)]:
        cells = []
        for r in runs:
            key = str(prior[r]) if w == "prior" else w
            cells.append(f"{dev[r][key]['対象語']:6.2f} / {dev[r][key]['文一致']:6.2f}")
        label = {"None": "行列なし", "prior": "dict_prior_weight"}.get(w, f"重み {w}")
        print(f"  {label:<20}" + "   ".join(cells))
    mean_dev = {w: statistics.mean(dev[r][str(w)]["対象語"] for r in runs) for w in SWEEP}
    chosen = max(SWEEP, key=lambda w: (mean_dev[w], -w))  # 同点なら小さい重み
    print(f"★dev で選んだ重み: {chosen}（seed 平均の対象語 {mean_dev[chosen]:.2f}%）")

    # --- 構成 ---
    configs = [
        Config(REF_NAME, "Uref", None),
        Config("U0", "U0", None),
        Config("U", "U0", 1.0),
        Config("U（補完 = 0）", "U0", 1.0, "zero"),
        Config("U（補完 = 中央値）", "U0", 1.0, "median"),
    ]
    for r in runs:
        configs += [
            Config(f"N0 [{r}]", r, None),
            Config(f"N [{r}]", r, prior[r]),
            Config(f"N dev [{r}]", r, chosen),
            Config(f"N（補完 = 0） [{r}]", r, prior[r], "zero"),
            Config(f"N（補完 = 中央値） [{r}]", r, prior[r], "median"),
            Config(f"N0 fp32 [{r}]", f"{r}-fp32", None),
            Config(f"N fp32 [{r}]", f"{r}-fp32", prior[r]),
        ]

    # --- benchmark ---
    items = load_joyo_benchmark(args.benchmark)
    if args.limit:
        items = items[: args.limit]
    print(f"\n=== benchmark {len(items):,} 件 ===")
    preds = run_items(items, dic, make_bases(), configs, matrix, "benchmark")
    preds[OJ_NAME] = openjtalk_preds(items)
    # ★INT8 の履歴依存の大きさ: dev を流した後のスコアラで N0 を測り直す
    preds.update(
        run_items(
            items,
            dic,
            dev_bases,
            [Config(f"N0 dev の後 [{r}]", r, None) for r in runs],
            matrix,
            "benchmark（dev の後の INT8）",
        )
    )
    bench = {name: bench_metrics(p, items) for name, p in preds.items()}

    # ★公式の物差し（jkyb-eval）. 自前の「対象語」は公式より 0.2〜0.6 pt 甘い（R-28）.
    #   解釈表の判定は、事前に登録したとおり自前の値で行う（ここは併記）.
    official = {}
    if not args.no_official:
        print("\n公式評価ツール（jkyb-eval）で測っています ...")
        official = evaluate_many(
            {name: [s for s, _ in p] for name, p in preds.items()},
            [it.key for it in items],
            args.benchmark,
            args.out_dir / "official",
        )

    # 3 分類の誤り分析（対象語を外した文）. 分類は文ごとに 1 回だけ
    category: dict[int, ErrorCategory] = {}
    errors: dict[str, dict[str, int]] = {}
    for name, p in preds.items():
        counts: Counter[str] = Counter()
        for k, ((_, span), it) in enumerate(zip(p, items, strict=True)):
            if hit(span, it):
                continue
            if k not in category:
                category[k] = classify_error(it.text, normalize_kana(it.reading), dic)
            counts[category[k].name] += 1
        errors[name] = dict(counts)

    # 対応のある比較: 行列で直った / 壊れた（対象語）
    def paired(a: str, b: str) -> dict[str, int]:
        ha = [hit(s, it) for (_, s), it in zip(preds[a], items, strict=True)]
        hb = [hit(s, it) for (_, s), it in zip(preds[b], items, strict=True)]
        return {
            "直った": sum(1 for x, y in zip(ha, hb, strict=True) if not x and y),
            "壊れた": sum(1 for x, y in zip(ha, hb, strict=True) if x and not y),
        }

    pairs = {"U0 -> U": paired("U0", "U")}
    for r in runs:
        pairs[f"N0 -> N [{r}]"] = paired(f"N0 [{r}]", f"N [{r}]")
        pairs[f"N0 -> N fp32 [{r}]"] = paired(f"N0 fp32 [{r}]", f"N fp32 [{r}]")
        pairs[f"U -> N [{r}]"] = paired("U", f"N [{r}]")

    # --- Hard Set（★run_hard_set.py と同じく、1 組のスコアラでカテゴリ順に流す） ---
    hard: dict[str, dict[str, float]] = {name: {} for name in preds}
    hs_bases = make_bases()
    for cat in HARD_SET_CATEGORIES:
        path = args.hard_set / f"{cat}.jsonl"
        if not path.exists():
            continue
        hs_items = load_joyo_benchmark(path)
        hs = run_items(hs_items, dic, hs_bases, configs, matrix, cat)
        hs[OJ_NAME] = openjtalk_preds(hs_items)
        for name, p in hs.items():
            n_hit = sum(hit(s, it) for (_, s), it in zip(p, hs_items, strict=True))
            hard[name][cat] = n_hit / len(hs_items) * 100

    # --- 表 ---
    def agg(prefix: str, metric: str, table: dict) -> tuple[float, float, list[float]]:
        vals = [table[f"{prefix} [{r}]"][metric] for r in runs]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return statistics.mean(vals), sd, vals

    n_prefixes = ["N0", "N", "N dev", "N（補完 = 0）", "N（補完 = 中央値）",
                  "N0 fp32", "N fp32", "N0 dev の後"]
    print("\n=== benchmark（対象語 / 文一致 / Sentence PER）===")
    for name in [REF_NAME, "U0", "U", "U（補完 = 0）", "U（補完 = 中央値）"]:
        m = bench[name]
        print(f"  {name:<34} {m['対象語']:6.2f}  {m['文一致']:6.2f}  {m['Sentence PER']:5.2f}")
    for prefix in n_prefixes:
        cells = []
        for metric in ("対象語", "文一致", "Sentence PER"):
            mean, sd, vals = agg(prefix, metric, bench)
            cells.append(f"{mean:6.2f} ± {sd:4.2f} ({' / '.join(f'{v:.2f}' for v in vals)})")
        print(f"  {prefix:<34} " + "  ".join(cells))
    m = bench[OJ_NAME]
    print(f"  {OJ_NAME:<34} {m['対象語']:6.2f}  {m['文一致']:6.2f}  {m['Sentence PER']:5.2f}")

    if official:
        print("\n=== ★公式の物差し（jkyb-eval。論文と同じ定義）: "
              "Accuracy / Target PER / Sentence PER ===")
        for name in [REF_NAME, "U0", "U", "U（補完 = 0）", "U（補完 = 中央値）"]:
            m = official[name]
            print(f"  {name:<34} {m.accuracy:6.2f}  {m.target_per:5.2f}  {m.sentence_per:5.2f}")
        for prefix in n_prefixes:
            cells = []
            for field in ("accuracy", "target_per", "sentence_per"):
                vals = [getattr(official[f"{prefix} [{r}]"], field) for r in runs]
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                cells.append(f"{statistics.mean(vals):6.2f} ± {sd:4.2f}")
            print(f"  {prefix:<34} " + "  ".join(cells))
        m = official[OJ_NAME]
        print(f"  {OJ_NAME:<34} {m.accuracy:6.2f}  {m.target_per:5.2f}  {m.sentence_per:5.2f}")

    shown = [REF_NAME, "U0", "U", *[f"{p} [{r}]" for p in ("N0", "N") for r in runs], OJ_NAME]
    print("\n=== 誤り 3 分類（対象語を外した文）===")
    for name in shown:
        print(f"  {name:<34} {errors[name]}")

    print("\n=== 対応のある比較（対象語）===")
    for k, v in pairs.items():
        print(f"  {k:<28} 直った {v['直った']:4d} / 壊れた {v['壊れた']:4d}")

    print("\n=== Hard Set（対象語）===")
    print("  " + " " * 34 + "".join(f"{c[:12]:>14}" for c in HARD_SET_CATEGORIES))
    for name in shown:
        cells = "".join(f"{hard[name].get(c, float('nan')):14.2f}" for c in HARD_SET_CATEGORIES)
        print(f"  {name:<34}{cells}")

    # --- ★解釈表（計画に事前登録した閾値. 書き換えない） ---
    u, u0, oj = bench["U"]["対象語"], bench["U0"]["対象語"], bench[OJ_NAME]["対象語"]
    n_mean = agg("N", "対象語", bench)[0]
    a = u - u0
    b = [bench[f"N [{r}]"]["対象語"] - bench[f"N0 [{r}]"]["対象語"] for r in runs]
    b32 = [bench[f"N fp32 [{r}]"]["対象語"] - bench[f"N0 fp32 [{r}]"]["対象語"] for r in runs]
    c = n_mean - u
    rows_hit: list[str] = []
    if a < 1.0:
        rows_hit.append("(i)")
    else:
        if n_mean >= oj - 0.5:
            rows_hit.append("(ii)")
        if abs(c) < 0.6:
            rows_hit.append("(iii)")
        if c >= 0.6 and n_mean < oj - 0.5:
            rows_hit.append("(iv)")
        if all(x < 0 for x in b):
            rows_hit.append("(v)")
    same_sign = all(x > 0 for x in b) or all(x < 0 for x in b)
    print("\n=== ★解釈表 ===")
    print(f"  A = U − U0 = {a:+.2f} pt（閾値 1.0）")
    print(f"  B = N − N0（seed ごと）= {', '.join(f'{x:+.2f}' for x in b)}"
          f" -> {'3 本とも同じ向き' if same_sign else '向きがそろわない'}")
    print(f"    FP32 での同じ差       = {', '.join(f'{x:+.2f}' for x in b32)}")
    print(f"  C = N − U = {c:+.2f} pt（閾値 0.6）/ N = {n_mean:.2f} / OJ − 0.5 = {oj - 0.5:.2f}")
    print(f"  ★当たる行: {', '.join(rows_hit) if rows_hit else 'どの行にも当たらない'}")

    # --- 保存 ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "n_benchmark": len(items),
        "median_cost": matrix.median,
        "dict_prior_weight": prior,
        "dev_path": str(dev_path),
        "dev": dev,
        "dev_chosen_weight": chosen,
        "benchmark": bench,
        "official": {name: vars(m) for name, m in official.items()},
        "errors": errors,
        "paired": pairs,
        "hard_set": hard,
        "judgement": {"A": a, "B": b, "B_fp32": b32, "C": c, "N": n_mean, "U": u, "U0": u0,
                      "OJ": oj, "rows": rows_hit},
    }
    (args.out_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (args.out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for name, p in preds.items():
            f.write(json.dumps({"config": name, "benchmark": p}, ensure_ascii=False) + "\n")
    print(f"\n保存: {args.out_dir}/results.json / predictions.jsonl")


if __name__ == "__main__":
    main()
