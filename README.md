# Do AI Traders Collude? Diagnosing algorithmic collusion from Q-learning to language models

Code, experiments and paper for a study of whether reinforcement-learning
informed traders in a repeated Kyle (1985) market collude (sustain low trading
with the threat of punishment) or merely under-trade because of a learning
bias ("over-pruning").

- **Paper:** `paper/main_v2.tex` (current: Q-learning, in-context and
  reasoning language models, RL-trained language models); `paper/main.tex` and
  `paper/main_ec.tex` are the earlier Q-learning-only versions. Figures and
  numbers are built from the result files by `paper/make_results.py`,
  `paper/make_llm_figures.py` and `paper/make_llm_tables.py`.
- **Experiment logs:** `results/NOTES.md` (Q-learning), `results/LLM_NOTES.md`
  (language models), plan in `results/PAPER_PLAN.md`

## Language-model traders

- `src/kylecollusion/llm_traders.py`: in-context traders (neutral prompts,
  paraphrase, framing and monitoring conditions; vLLM, OpenAI and scripted
  backends with spend caps) and the paired deviation test with shared
  sampling seeds.
- `src/kylecollusion/grpo.py`: GRPO self-play with LoRA (common random numbers
  within groups, discounted return-to-go, gamma = 0 as myopic control) and a
  scripted trigger rival as a positive control.
- Runners: `experiments/llm_pilot.py` (one condition), `experiments/grpo_train.py`,
  `experiments/grpo_audit.sh`, `experiments/llm_*.sh`; `experiments/status.py`
  for monitoring; `experiments/llm_tabulate.py` for quick tables.

## Main ideas

- **Diagnostics** that separate punishment from pruning: a paired
  rival-deviation test, the Dou et al. (2025) noise-shock test, a myopic
  placebo (gamma = 0 learners cannot punish), a no-memory control, and
  learning-rate sensitivity.
- **Detectability:** in the standard Kyle market a best-response deviation
  moves order flow by only (I-1)/(2I) noise standard deviations per unit of
  v / sigma_v, independent of noise volume; with many information-insensitive
  investors it is hundreds of standard deviations (`theory.deviation_gap`).
- **Mechanism:** with no rival at all, constant-step Q-learning under-trades
  by about 0.91 sqrt(alpha); counterfactual updates remove the bias.
- **Findings:** no punishment in any configuration (even with perfect
  monitoring, or deviations 625 noise sd large); myopic placebos and an
  uninformative memory reproduce the "collusive" outcome; with passive traders
  learners trade half their Nash intensity although collusion would mean
  trading more; reducing transparency changes nothing; counterfactual updating
  restores competition; the collusion index rises steadily with the step size;
  a myopic shared-table learner shows the noise-shock "trigger" signature,
  which comes from never-visited states.

- **Validation:** the same diagnostics find punishment in the Calvano et al.
  (2020) logit-Bertrand game (`bertrand.py`), and flag its memoryless variant
  (high prices, no punishment) as a learning artefact. They are also applied
  to competing Q-learning dealers under adverse selection (`quotes.py`).
- **Sustainability:** Green-Porter style bounds (`sustain.py`) on how much
  collusion trigger strategies can sustain given how visible deviations are.
- **Audit protocol:** the diagnostics as a five-step procedure (paper, Sec. 6).

## Model

Each period a value v is drawn; I learning informed traders (plus optional
passive Nash traders) submit orders; noise traders add u ~ N(0, sigma_u^2);
information-insensitive investors demand z = -xi (p - v_bar); the market maker
prices p = v_bar + lambda y with lambda = (theta * lambda_B + xi) / (theta + xi^2)
and re-estimates lambda_B from exponentially weighted moments. Exact Nash and
cartel benchmarks are in `theory.py` (closed form at xi = 0, numerical for
xi > 0; they reproduce Dou et al.'s published values).

Configurable pieces (`MarketConfig`, CLI flags in `run.py`):

| flag | meaning |
|---|---|
| `--memory none/flow/residual/orders/price/random` | what traders remember from last period (`random`: uninformative placebo state, `--n-random-states`) |
| `--price-bins noise/grid` | price state binned in noise-sd units, or over the range the order grid can produce (Dou et al. style) |
| `--grid-mode wide/bracket` | shared symmetric order grid, or Dou et al.'s per-value cartel-to-Nash bracket |
| `--xi`, `--theta` | information-insensitive investors and market-maker weights |
| `--n-passive`, `--disclosure-noise`, `--tick`, `--order-cap` | market-design interventions |
| `--algo q/dqn/ppo`, `--gamma`, `--agent-kwargs` | learner (Q options: `alpha`, `beta_decay`, `alpha_schedule`, `update` = taken/counterfactual, `shared`) |
| `--impulse-reps`, `--shock-devs` | rival-deviation test and noise-shock test |
| `--checkpoint` | resumable training |
| `--engine numba` | compiled training loop for tabular Q (10-20x faster; same model, different random draws) |

## Reproducing

```bash
pip install -e ".[dev]" matplotlib numba
pytest                                       # 80 tests
./experiments/run_all.sh                     # all sweeps; resumable after interruption
PYTHONPATH=src python experiments/mechanism.py            # single trader, step size
PYTHONPATH=src python experiments/mechanism.py --gamma    # single trader, discount factor
PYTHONPATH=src python experiments/rare_states.py          # off-path states and shock responses
PYTHONPATH=src python experiments/bertrand_validation.py                   # Calvano et al. validation
PYTHONPATH=src python experiments/bertrand_validation.py --market quotes    # dealers
PYTHONPATH=src python experiments/sustain_curves.py                        # sustainability bounds
python paper/make_results.py                 # figures + numbers.tex
cd paper && latexmk -pdf main.tex           # working paper
cd paper && latexmk -pdf main_ec.tex        # EC submission (anonymous, 18-page body, EC'26 style)
```

Experiment specs are in `experiments/grids/*.json`; `experiments/exp4.sh`
holds the perfect-monitoring runs. Everything runs on CPU: the simulator
advances hundreds of independent markets per numpy step (Q-learning) or as
batched per-agent networks (DQN, PPO).

## Layout

```
src/kylecollusion/
  theory.py        benchmarks, deviation signal-to-noise
  sustain.py       trigger-strategy sustainability bounds
  fast.py          numba training engine
  bertrand.py      Calvano et al. (2020) logit Bertrand with Q-learning
  quotes.py        competing dealers under adverse selection
  market.py        batched repeated Kyle market
  agents/          tabular Q (constant/decaying step, taken/counterfactual updates), DQN, PPO
  diagnostics.py   deviation test, noise-shock test, convergence statistics
  metrics.py       collusion indices and per-session outcomes
  run.py           train, evaluate, test; JSON output
experiments/       sweep runner, grids, queue, mechanism experiment, tabulation
results/           result JSON per run, NOTES.md
paper/             LaTeX sources, figures, make_results.py
tests/             theory, market, agents, diagnostics
```
