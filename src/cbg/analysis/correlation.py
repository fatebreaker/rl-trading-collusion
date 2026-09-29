"""Cross-benchmark transfer analysis (RQ1, RQ2, RQ3).

Pure-python + statistics only, so it runs in CI without numpy/scipy. Swap in
scipy for the paper-grade CIs if desired; the estimators here are correct for
the small model x benchmark matrices we work with.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

from ..core.task import TaskResult


def _ranks(xs: list[float]) -> list[float]:
    """Average-rank transform (ties share the mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation. Returns nan for degenerate input."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    mra, mrb = mean(ra), mean(rb)
    num = sum((x - mra) * (y - mrb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - mra) ** 2 for x in ra) * sum((y - mrb) ** 2 for y in rb))
    return num / den if den else float("nan")


def per_model_scores(results: list[TaskResult]) -> dict[str, dict[str, float]]:
    """model -> benchmark -> mean score."""
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        bucket[r.model][r.benchmark.value].append(r.score)
    return {
        m: {b: mean(v) for b, v in benchs.items()}
        for m, benchs in bucket.items()
    }


def transfer_matrix(results: list[TaskResult]) -> dict[tuple[str, str], float]:
    """RQ1: Spearman ρ of per-model scores between every benchmark pair.

    High ρ => models rank consistently across benchmarks (evidence of one
    shared capability). Low/near-zero ρ => the axes are largely independent.
    """
    scores = per_model_scores(results)
    benchmarks = sorted({b for m in scores.values() for b in m})
    models = sorted(scores)
    out: dict[tuple[str, str], float] = {}
    for i, b1 in enumerate(benchmarks):
        for b2 in benchmarks[i + 1 :]:
            paired = [
                (scores[m][b1], scores[m][b2])
                for m in models
                if b1 in scores[m] and b2 in scores[m]
            ]
            if len(paired) >= 2:
                out[(b1, b2)] = spearman([p[0] for p in paired], [p[1] for p in paired])
    return out


def contamination_gap(results: list[TaskResult], cutoff_iso: str) -> dict[str, float]:
    """RQ3: score(pre-cutoff disclosure) - score(post-cutoff), per model.

    Positive gap is consistent with contamination helping on vulns the model
    could have seen in pretraining. Interpreted as a lower bound.
    """
    from datetime import date

    cut = date.fromisoformat(cutoff_iso + "-01" if len(cutoff_iso) == 7 else cutoff_iso)
    pre: dict[str, list[float]] = defaultdict(list)
    post: dict[str, list[float]] = defaultdict(list)
    for r in results:
        d = r.grader_detail.get("disclosure_date")
        if not d:
            continue
        (pre if date.fromisoformat(d) < cut else post)[r.model].append(r.score)
    return {
        m: mean(pre[m]) - mean(post[m])
        for m in pre
        if pre[m] and post.get(m)
    }
