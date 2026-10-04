"""LLM-trader plumbing, checked with scripted 'models' whose behaviour is known."""

import re

import numpy as np
import pytest

from kylecollusion.llm_traders import (
    LLMTraderConfig, LLMTraders, ScriptedBackend, deviation_test, fixed_lambda_benchmarks,
    parse_response, system_prompt,
)
from kylecollusion.market import KyleMarket, MarketConfig


def _value(conv):
    return float(re.search(r"This period's value: V = ([-+]\d+\.\d+)", conv[-1]["content"]).group(1))


def _last_row(conv):
    """(V, own order, total flow) of the most recent history row, or None."""
    rows = [l for l in conv[-1]["content"].splitlines() if re.match(r"^\d+ \| ", l)]
    if not rows:
        return None
    f = [float(c) for c in rows[-1].split(" | ")]
    return f[1], f[2], f[-3]


def scripted(base, slope=0.0, noise=0.0):
    """Order = (base + slope * surprise) * V, surprise = sign(V_prev) * (flow - own order).

    slope = 0 ignores history (cannot punish); slope > 0 trades harder after
    others traded more than usual, a crude punishment rule. `noise` adds
    seed-determined jitter, like sampling at positive temperature."""

    def fn(conv, seed):
        v = _value(conv)
        row = _last_row(conv)
        surprise = 0.0 if row is None else np.sign(row[0]) * (row[2] - row[1])
        eps = noise * np.random.default_rng(seed).normal()
        return f'{{"notes": "ok", "order": {(base + slope * surprise) * v + eps:.6f}}}'

    return ScriptedBackend(fn)


def _market(I=2, S=200, seed=1):
    env = KyleMarket(MarketConfig(n_informed=I, mm_fixed=True, memory="flow"), S, seed=seed)
    env.reset()
    return env


def test_parse_response():
    assert parse_response('{"notes": "hi", "order": -1.25}') == (-1.25, "hi")
    assert parse_response('<think>{"order": 9}</think>\n{"order": 0.5}')[0] == 0.5
    assert parse_response('Sure: {"order": "2.0"}')[0] == 2.0
    assert parse_response('```json\n{"notes": "x", "order": 1e-1}\n```')[0] == pytest.approx(0.1)
    assert parse_response('blah "order": 3 blah')[0] == 3.0
    assert parse_response("no idea")[0] is None
    assert parse_response('{"order": NaN}')[0] is None


def test_prompt_is_neutral_and_conditions_differ():
    vals = np.array([-1.0, 0.0, 1.0])
    base = system_prompt(vals, 2, LLMTraderConfig())
    for word in ("collu", "compet", "cooperat", "cartel", "punish", "lambda"):
        assert word not in base.lower()
    assert "One other trader" in base
    assert "No other trader" in system_prompt(vals, 1, LLMTraderConfig())
    assert "current period" in system_prompt(vals, 2, LLMTraderConfig(objective="myopic"))
    assert "see what the other trader ordered" in system_prompt(vals, 2, LLMTraderConfig(show_rival=True))


def test_nash_script_scores_nash():
    env = _market()
    fb = fixed_lambda_benchmarks(2, env.bench.lam_nash, 1.0)
    # frozen at the Nash lambda, the stage Nash is the Kyle Nash
    assert fb["beta_nash"] == pytest.approx(env.bench.beta_nash)
    tr = LLMTraders(env, LLMTraderConfig(history=5), seed=0)
    be = scripted(fb["beta_nash"])
    xs, vs = [], []
    for _ in range(30):
        x = tr.act(env, be)
        profit, _, info = env.step_orders(x)
        tr.record(info, profit)
        xs.append(x.sum(1))
        vs.append(info["v"])
    xs, vs = np.array(xs), np.array(vs)
    beta = (xs * vs).sum() / (vs * vs).sum()
    assert beta == pytest.approx(fb["agg_nash"], rel=5e-3)  # prompts round V to 2 decimals
    assert tr.n_fail == 0
    assert len(tr.hist[0][0]) == 30


def _dev(slope, noise=0.0):
    env = _market()
    tr = LLMTraders(env, LLMTraderConfig(history=5), seed=0)
    be = scripted(env.bench.beta_coll, slope=slope, noise=noise)
    for _ in range(5):
        x = tr.act(env, be)
        profit, _, info = env.step_orders(x)
        tr.record(info, profit)
    return deviation_test(env, tr, be, events=4, gap=3, horizon=4)


def test_memoryless_traders_do_not_react():
    d = _dev(slope=0.0, noise=0.05)  # seeded noise is shared across branches
    assert d["n_events"] > 0
    assert np.allclose(d["d_beta_rival"], 0.0)
    assert d["d_beta_dev"][0] > 0  # the deviation trades harder
    assert d["cum_gain_dev"] > 0  # and with no reaction it pays


def test_reactive_traders_are_detected():
    d = _dev(slope=0.3, noise=0.05)
    assert d["d_beta_rival"][0] == pytest.approx(0.0)  # rivals react after, not during
    assert d["d_beta_rival"][1] - d["d_beta_rival_ci95"][1] > 0


def test_solo_market():
    env = _market(I=1, S=20)
    fb = fixed_lambda_benchmarks(1, env.bench.lam_nash, 1.0)
    assert fb["agg_nash"] == pytest.approx(fb["agg_coll"])  # a monopolist has no rival
    tr = LLMTraders(env, LLMTraderConfig(), seed=0)
    assert "No other trader" in tr.system
    x = tr.act(env, scripted(1.0))
    assert x.shape == (20, 1)


def test_shift_deviation_is_detected_and_visible():
    env = _market()
    tr = LLMTraders(env, LLMTraderConfig(history=5), seed=0)
    be = scripted(env.bench.beta_coll, slope=0.3, noise=0.05)
    d = deviation_test(env, tr, be, events=3, gap=2, horizon=3, mode="shift", scale=2.0)
    assert d["d_beta_dev"][0] > 0  # shift always trades harder in the direction of v
    assert d["d_beta_rival"][1] - d["d_beta_rival_ci95"][1] > 0


def test_vague_framing_is_identical_with_and_without_rival():
    vals = np.array([-1.0, 0.0, 1.0])
    cfg = LLMTraderConfig(rival_info="vague")
    assert system_prompt(vals, 1, cfg) == system_prompt(vals, 2, cfg)
    assert "some of whom may also know V" in system_prompt(vals, 2, cfg)


def test_paraphrase_variant():
    vals = np.array([-1.0, 0.0, 1.0])
    b = LLMTraderConfig(prompt_variant="b")
    pa, pb = system_prompt(vals, 2, LLMTraderConfig()), system_prompt(vals, 2, b)
    assert pa != pb and "another participant" in pb and "+1.00" in pb
    for word in ("collu", "compet", "cooperat", "cartel", "punish", "lambda"):
        assert word not in pb.lower()
    assert "Nobody else" in system_prompt(vals, 1, b)
    assert system_prompt(vals, 1, LLMTraderConfig(prompt_variant="b", rival_info="vague")) == \
        system_prompt(vals, 2, LLMTraderConfig(prompt_variant="b", rival_info="vague"))
    # the scripted trader still finds V and the history rows in the paraphrased user prompt
    env = _market(S=4)
    tr = LLMTraders(env, LLMTraderConfig(prompt_variant="b", history=3), seed=0)
    be = ScriptedBackend(lambda conv, seed: '{"order": %s}' % re.search(
        r"V this round = ([-+]\d+\.\d+)", conv[-1]["content"]).group(1))
    for _ in range(3):
        x = tr.act(env, be)
        pi, _, info = env.step_orders(x)
        tr.record(info, pi)
    assert tr.n_fail == 0
    assert "Your last 3 rounds" in tr.user_prompt(0, 0, 1.0)


def test_instructions_only_when_given():
    vals = np.array([-1.0, 0.0, 1.0])
    assert system_prompt(vals, 2, LLMTraderConfig()).endswith("<number>}")
    p = system_prompt(vals, 2, LLMTraderConfig(instructions="Rule: X."))
    assert p.endswith("\n\nRule: X.")


def test_raw_capture_and_triopoly_prompt():
    env = _market(I=3, S=4)
    tr = LLMTraders(env, LLMTraderConfig(history=3), seed=0)
    assert "2 other traders also learn V" in tr.system
    tr.raw = []
    x = tr.act(env, scripted(0.5))
    assert len(tr.raw) == 4 * 3 and tr.raw[0][0] == 0
    assert x.shape == (4, 3)


def test_openai_snapshot_pricing():
    from kylecollusion.llm_traders import OpenAIBackend
    b = OpenAIBackend("gpt-5.4-nano-2026-03-17")
    assert b._cost(1e6, 0, 0) == pytest.approx(0.20)
    with pytest.raises(ValueError):
        OpenAIBackend("gpt-4o")


def test_parse_after_closing_think_tag_only():
    # R1-distill templates open <think> themselves: only the closing tag appears
    text = 'draft {"order": 9.0} ...</think>\n\n{"notes": "ok", "order": 0.5}'
    assert parse_response(text) == (0.5, "ok")
    assert parse_response('reasoning {"order": 9.0} </think> no answer')[0] is None
