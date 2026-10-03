"""GRPO helpers: common random numbers within groups, advantages."""

import numpy as np
import pytest

from kylecollusion.grpo import GroupRNG, group_advantages
from kylecollusion.market import KyleMarket, MarketConfig


def test_sessions_in_a_group_share_values_and_noise():
    env = KyleMarket(MarketConfig(mm_fixed=True, memory="flow"), 12, seed=5)
    env.rng = GroupRNG(env.rng, 4)
    env.reset()
    for _ in range(5):
        v_idx = env.v_idx.reshape(3, 4)
        assert (v_idx == v_idx[:, :1]).all()
        x = np.random.default_rng(0).normal(size=(12, 2))  # different orders per session
        _, _, info = env.step_orders(x)
    # same values and noise: flow differences are exactly the order differences
    y = info["y"].reshape(3, 4)
    xs = x.sum(1).reshape(3, 4)
    assert np.allclose(y - xs, (y - xs)[:, :1])
    with pytest.raises(ValueError):
        env.rng.integers(3, size=(12, 2))


def test_group_advantages():
    T, S, I, G = 3, 8, 2, 4
    rng = np.random.default_rng(1)
    profit = rng.normal(size=(T, S, I))
    adv = group_advantages(profit, 0.0, G)
    # myopic credit: centred within group, same (t, trader slot)
    g = adv.reshape(T, S // G, G, I)
    assert np.allclose(g.sum(2), 0.0)
    c = profit.reshape(T, S // G, G, I) - profit.reshape(T, S // G, G, I).mean(2, keepdims=True)
    assert np.allclose(adv, (c / c.std()).reshape(T, S, I))
    # identical outcomes within a group carry no signal
    assert np.allclose(group_advantages(np.ones((T, S, I)), 0.9, G), 0.0)
    # with gamma > 0 a later profit gap is credited to earlier periods
    p = np.zeros((T, S, I))
    p[2, 0, 0] = 1.0
    a = group_advantages(p, 0.5, G)
    assert a[0, 0, 0] > 0 and a[0, 0, 0] < a[2, 0, 0]
    assert np.allclose(a[:, :, 1], 0.0)


class FakePolicy:
    """Stands in for PolicyBackend: order = mult x V, with fake token ids."""

    def __init__(self, mult):
        import re
        self.mult, self.re, self.records = mult, re, None

    def generate(self, convs, seeds, temperature, max_tokens):
        texts = []
        for c in convs:
            v = float(self.re.search(r"V = ([-+]\d+\.\d+)", c[-1]["content"]).group(1))
            texts.append('{"order": %.4f}' % (self.mult * v))
        if self.records is not None:
            self.records.append([([1, 2, 3], [4, 5]) for _ in convs])
        return texts


def _trigger_cfg(**kw):
    from kylecollusion.grpo import GRPOConfig
    return GRPOConfig(rival="trigger", groups=2, group_size=3, periods=12, **kw)


def test_trigger_rival_punishes_deviations_only():
    from kylecollusion.grpo import rollout
    cfg = _trigger_cfg()
    s_dev, st_dev = rollout(FakePolicy(0.8), cfg, seed=3)
    assert st_dev["punished_share"] > 0.3
    assert len(s_dev) == cfg.periods * cfg.groups * cfg.group_size  # policy trader only
    coop = st_dev["coop"]
    _, st_coop = rollout(FakePolicy(coop), cfg, seed=3)
    assert st_coop["punished_share"] == 0.0
    assert st_coop["policy_intensity"] == pytest.approx(coop, rel=1e-2)
    # cooperating beats deviating once punishment is counted
    assert st_coop["policy_profit"] > st_dev["policy_profit"]
    # the one-shot best response to cooperation is to trade more
    assert st_dev["policy_br_to_coop"] > coop
