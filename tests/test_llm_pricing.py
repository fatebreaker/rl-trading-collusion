import re

import numpy as np

from kylecollusion.llm_pricing import (
    LLMPricers, PricingConfig, PricingMarket, best_response, deviation_test, parse_price,
    pricing_benchmarks, run_period,
)
from kylecollusion.llm_traders import ScriptedBackend


def test_benchmarks_scale_with_currency_unit():
    b1, b10 = pricing_benchmarks(1.0), pricing_benchmarks(10.0)
    assert abs(b1["p_nash"] - 1.473) < 1e-3 and abs(b1["p_mono"] - 1.925) < 1e-3
    assert abs(b10["p_nash"] - 10 * b1["p_nash"]) < 1e-9
    assert abs(b10["pi_nash"] - 10 * b1["pi_nash"]) < 1e-9


def test_best_response_to_nash_is_nash_and_scales():
    for k in (1.0, 10.0):
        env = PricingMarket(PricingConfig(scale=k), 1)
        pn = env.bench["p_nash"]
        assert abs(best_response(np.array([pn]), env.bcfg)[0] - pn) < 2e-3 * k


def test_parse_price():
    assert parse_price('{"notes": "hold", "price": 1.85}') == (1.85, "hold")
    assert parse_price('{"price": "$2.10"}')[0] == 2.10
    assert parse_price("no price here")[0] is None


def _rival_prices(conv):
    rows = [l for l in conv[1]["content"].splitlines() if re.match(r"^\d+ \|", l)]
    return [float(r.split(" | ")[2]) for r in rows[-2:]]


def _trigger_backend(bench):
    agreed, punish = bench["p_mono"], bench["p_nash"]
    tol = 0.25 * (bench["p_mono"] - bench["p_nash"])

    def fn(conv, seed):
        broke = any(p < agreed - tol for p in _rival_prices(conv))
        return '{"price": %.4f}' % (punish if broke else agreed)
    return ScriptedBackend(fn)


def test_deviation_test_detects_scripted_trigger_strategy():
    cfg = PricingConfig(notes=False)
    env = PricingMarket(cfg, 8)
    pr = LLMPricers(env, cfg, seed=1)
    be = _trigger_backend(env.bench)
    for _ in range(4):
        run_period(env, pr, be)
    res = deviation_test(env, pr, be, events=2, gap=2, horizon=4)
    assert res["deviation_size"] > 0  # the best response undercuts the agreed price
    assert res["rival_aggression"][1] > 0.9  # the rival drops to Nash after the deviation
    assert res["cum_gain_dev"] < 0  # and the deviation does not pay


def test_deviation_test_finds_nothing_without_strategy():
    cfg = PricingConfig(notes=False)
    env = PricingMarket(cfg, 6)
    pr = LLMPricers(env, cfg, seed=2)
    be = ScriptedBackend(lambda conv, seed: '{"price": 1.80}')
    for _ in range(3):
        run_period(env, pr, be)
    res = deviation_test(env, pr, be, events=2, gap=2, horizon=4)
    assert all(abs(x) < 1e-12 for x in res["rival_aggression"])
