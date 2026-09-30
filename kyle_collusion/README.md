# kylecollusion: can market design stop AI trading collusion?

Learning informed traders in a repeated Kyle (1985) market, the conditions under
which they learn to collude, and which market-design interventions break it —
tested across tabular Q-learning, DQN and PPO.

Motivation: Dou, Goldstein & Ji (NBER w34054, 2025) show that RL-powered
informed speculators can learn to collude without communicating, cutting
liquidity and price informativeness. The regulator's question is what to do
about it. Collusion is known to depend on the learning algorithm (Deng et al.,
2024), so an intervention is only credible if it works across algorithms.

## Model

Each period (repeated forever, with discount factor gamma for the learners):

| | |
|---|---|
| value | `v` drawn from a 5-point equiprobable grid with variance `sigma_v^2`, seen by informed traders |
| learners | `I` informed traders each pick an order `x_i` from a 31-point grid |
| passive | `P` optional non-learning informed traders playing the Nash strategy |
| noise | `u ~ N(0, sigma_u^2)` |
| flow | `y = sum x_i + passive + u` |
| price | `p = E[v] + lambda (y - E[y])`; the market maker re-estimates `lambda = Cov(v,y)/Var(y)` with exponentially weighted moments (half-life 2000 periods) |
| profit | `(v - p) x_i` |

What a learner remembers from the last period (`--memory`):
- `none`: nothing. It cannot punish, so any under-trading is learning bias. **This is the control.**
- `flow`: binned aggregate order flow `y`.
- `residual`: the previous value and the binned `y - own x` (others + noise).

Benchmarks (`theory.py`, exact under linear pricing):
- **Nash**: symmetric Kyle equilibrium with `n = I + P` insiders: `beta = sigma_u / (sqrt(n) sigma_v)`.
- **Collusive**: learners jointly act as one monopolist insider facing a rational
  market maker: aggregate `B_L = sqrt(B_P^2 + sigma_u^2 / sigma_v^2)`.

Outcomes are reported as a Calvano-style index `Delta = (m - m_Nash)/(m_coll - m_Nash)`
(0 = competition, 1 = full collusion) for profit, aggregate trading intensity and
price informativeness, plus diagnostics: `order_r2` (share of order variance
explained by `v`) and `policy_change` (share of the greedy strategy that changed
over the last 100k training steps; Calvano's convergence criterion in spirit).

Model notes worth knowing:
- With passive Nash traders, `I = P + 1` makes the collusive and Nash benchmarks
  coincide exactly (no gain from colluding), so avoid that ratio. See `test_theory.py`.
- Q-learning under-trades even alone, because noisy large-order payoffs get stuck
  with unlucky low estimates. This is why the memoryless control is required.
  See `test_q_learning_undertrades_under_reward_noise`.

## Market-design interventions (config flags)

`--sigma-u` noise volume · `--n-passive` non-learning informed traders ·
`--tick` price grid · `--order-cap` position limit · `--disclosure-noise`
noise on the flow traders observe (transparency) · `--mm-fixed` freeze lambda (ablation).

## Running

```bash
pip install -e ".[dev]"
pytest                                   # 47 tests: theory, market mechanics, agents
PYTHONPATH=src python -m kylecollusion.run --algo q --memory flow --sessions 200 \
    --steps 6000000 --agent-kwargs '{"beta_decay": 1e-6}' --out results/q_flow.json
PYTHONPATH=src python experiments/sweep.py experiments/grids/<grid>.json --workers 4
```

Sessions are simulated in parallel inside one process (numpy for Q-learning,
batched per-agent networks for DQN/PPO), so everything runs on CPU.
Rough cost on 4 cores, 200 sessions: Q-learning 1.5M steps ≈ 6 min.
DQN and PPO are about 20x and 12x slower per step.

## Status and caveats

- Built from the standard Kyle model. **Not yet checked against Dou, Goldstein &
  Ji's exact specification** (their PDF was not reachable from this environment);
  align value distribution, market-maker learning rule, state space and
  hyperparameters with theirs before making comparisons.
- Results so far are preliminary; see `results/NOTES.md`.
