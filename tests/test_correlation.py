"""Tests for the transfer-analysis estimators — the paper's core computation."""

from datetime import date

from cbg.analysis.correlation import (
    contamination_gap,
    per_model_scores,
    spearman,
    transfer_matrix,
)
from cbg.core.task import Benchmark, TaskResult


def _r(model, bench, score, disclosure=None):
    detail = {"disclosure_date": disclosure} if disclosure else {}
    return TaskResult(
        task_id=f"{bench.value}-{model}-{score}",
        benchmark=bench,
        model=model,
        solved=score >= 0.5,
        score=score,
        steps_used=1,
        submission=None,
        grader_detail=detail,
    )


def test_spearman_monotonic():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_per_model_scores_averages():
    res = [_r("m1", Benchmark.CYBERGYM, 1.0), _r("m1", Benchmark.CYBERGYM, 0.0)]
    assert per_model_scores(res)["m1"]["cybergym"] == 0.5


def test_transfer_matrix_perfect_agreement():
    # Model ranking identical on both benchmarks -> rho == 1.0
    res = [
        _r("m1", Benchmark.CYBERGYM, 0.9), _r("m1", Benchmark.CYBENCH, 0.8),
        _r("m2", Benchmark.CYBERGYM, 0.5), _r("m2", Benchmark.CYBENCH, 0.4),
        _r("m3", Benchmark.CYBERGYM, 0.1), _r("m3", Benchmark.CYBENCH, 0.2),
    ]
    tm = transfer_matrix(res)
    assert tm[("cybench", "cybergym")] == 1.0


def test_contamination_gap_sign():
    # pre-cutoff solved, post-cutoff failed -> positive gap
    res = [
        _r("m1", Benchmark.CYBERGYM, 1.0, disclosure="2024-01-01"),
        _r("m1", Benchmark.CYBERGYM, 0.0, disclosure="2026-06-01"),
    ]
    gap = contamination_gap(res, "2025-06")
    assert gap["m1"] == 1.0
