"""Smoke and sanity tests for the learning agents."""

import numpy as np
import pytest
import torch

from kylecollusion.agents.batched_mlp import BatchedMLP, clip_grad_norm_per_net
from kylecollusion.agents.tabular_q import TabularQ
from kylecollusion.market import KyleMarket, MarketConfig
from kylecollusion.run import make_agent, parse_args, run


def test_batched_mlp_networks_are_independent():
    torch.manual_seed(0)
    net = BatchedMLP(3, 4, 2, hidden=8)
    x = torch.randn(3, 5, 4)
    net(x)[1].sum().backward()  # loss touches net 1 only
    for p in net.parameters():
        assert p.grad[0].abs().sum() == 0
        assert p.grad[2].abs().sum() == 0
        assert p.grad[1].abs().sum() > 0


def test_per_net_clipping_leaves_small_grads_alone():
    net = BatchedMLP(2, 3, 1, hidden=4)
    x = torch.randn(2, 7, 3)
    out = net(x)
    (out[0].sum() * 1e4 + out[1].sum() * 1e-4).backward()
    before = [p.grad[1].clone() for p in net.parameters()]
    clip_grad_norm_per_net(net, 1.0)
    for p, b in zip(net.parameters(), before):
        assert torch.allclose(p.grad[1], b)
    norm0 = sum(p.grad[0].pow(2).sum() for p in net.parameters()).sqrt()
    assert norm0 == pytest.approx(1.0, rel=1e-3)


def _train_single_trader(noise_scale: float, alpha: float, steps: int, sessions: int = 16):
    """I=1, memoryless, lambda frozen at the monopoly value, gamma=0.

    The optimal order is v / (2 lambda). `noise_scale` scales the noise-trader
    volume actually drawn, while lambda stays at its sigma_u=1 value, so the
    optimum is unchanged and only reward noise varies.
    """
    from dataclasses import replace

    cfg = MarketConfig(n_informed=1, memory="none", mm_fixed=True, n_actions=41)
    env = KyleMarket(cfg, sessions, seed=0)
    env.cfg = replace(cfg, sigma_u=cfg.sigma_u * noise_scale)
    agent = TabularQ(env, alpha=alpha, gamma=0.0, beta_decay=1e-4, seed=1)
    obs = env.reset()
    for t in range(steps):
        a = agent.act(obs, t)
        r, obs2, _ = env.step(a)
        agent.observe(obs, a, r, obs2, t)
        obs = obs2
    greedy = env.grid[agent.greedy_policy()[:, 0, 0]]  # (S, n_values)
    target = env.values / (2 * env.bench.lam_nash)
    return env, greedy, target


def test_single_trader_q_learns_best_response_without_noise():
    """Deterministic rewards isolate the update mechanics: must hit the optimum."""
    env, greedy, target = _train_single_trader(noise_scale=0.0, alpha=0.5, steps=40_000)
    step = env.grid[1] - env.grid[0]
    assert (np.abs(greedy - target) <= 0.5 * step + 1e-9).all(), (greedy[0], target)


def test_q_learning_undertrades_under_reward_noise():
    """Documents a real learning bias, not a bug.

    With noisy profits, rarely-chosen large orders can get stuck with unlucky
    low Q-estimates, so greedy play shrinks toward smaller orders even with no
    rival to collude with. Any collusion result has to be measured against this
    (the memoryless control does exactly that).
    """
    env, greedy, target = _train_single_trader(noise_scale=1.0, alpha=0.1, steps=60_000)
    hi = np.argmax(target)
    assert np.median(greedy[:, hi]) < target[hi]


@pytest.mark.parametrize("algo", ["q", "dqn", "ppo"])
def test_run_end_to_end(algo):
    args = parse_args(
        ["--algo", algo, "--sessions", "4", "--steps", "1200", "--eval-steps", "300"]
        + (["--agent-kwargs", '{"learning_starts": 200}'] if algo == "dqn" else [])
        + (["--agent-kwargs", '{"rollout_len": 64}'] if algo == "ppo" else [])
    )
    res = run(args)
    for k in ("profit", "agg_intensity", "delta_profit", "delta_intensity"):
        assert np.isfinite(res["summary"][k]["mean"])
    assert len(res["per_session"]["profit"]) == 4


def test_make_agent_rejects_unknown():
    env = KyleMarket(MarketConfig(), 2)
    with pytest.raises(ValueError):
        make_agent("sarsa", env, 0)


def test_visit_decay_schedule():
    env = KyleMarket(MarketConfig(), 2, seed=0)
    with pytest.raises(ValueError):
        TabularQ(env, alpha_schedule="linear")
    agent = TabularQ(env, alpha=0.2, alpha_schedule="visits", alpha_kappa=10, alpha_power=1.0)
    obs = env.reset()
    a = np.zeros((2, 2), dtype=int)
    idx = (agent._si, agent._ii, obs["s"], obs["v_idx"][:, None], a)
    r = np.ones((2, 2))
    # Repeated identical updates of one entry: step n uses alpha / (1 + n/10).
    q0 = agent.Q[idx].copy()
    target = r + agent.gamma * agent._rows(obs).max(-1)
    agent.observe(obs, a, r, obs, 0)
    assert np.allclose(agent.Q[idx], q0 + 0.2 * (target - q0))
    assert (agent.visits[idx] == 1).all()
    for _ in range(9):
        agent.observe(obs, a, r, obs, 0)
    assert (agent.visits[idx] == 10).all()


def test_visit_decay_still_finds_best_response():
    from dataclasses import replace

    cfg = MarketConfig(n_informed=1, memory="none", mm_fixed=True, n_actions=41)
    env = KyleMarket(cfg, 8, seed=0)
    env.cfg = replace(cfg, sigma_u=0.0)
    agent = TabularQ(env, alpha=0.5, gamma=0.0, beta_decay=1e-4, alpha_schedule="visits", seed=1)
    obs = env.reset()
    for t in range(40_000):
        a = agent.act(obs, t)
        r, obs2, _ = env.step(a)
        agent.observe(obs, a, r, obs2, t)
        obs = obs2
    greedy = env.grid[agent.greedy_policy()[:, 0, 0]]
    target = env.values / (2 * env.bench.lam_nash)
    step = env.grid[1] - env.grid[0]
    assert (np.abs(greedy - target) <= 0.5 * step + 1e-9).all()
