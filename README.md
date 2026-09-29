# cbg — Cross-Benchmark Generalization of LLM Security Agents

A research harness for measuring whether LLM agents have a single transferable
"security-reasoning" capability, or whether benchmark scores are dominated by
scaffold fit, harness quirks, and training-data contamination.

We run **one fixed agent scaffold** across **CyberGym**, **Cybench / NYU-CTF**,
and **CVE-Bench**, and analyze how performance transfers.

See [`docs/PROPOSAL.md`](docs/PROPOSAL.md) for the full research design,
research questions (RQ1–RQ4), method, threats to validity, and milestones.

## Layout

```
src/cbg/
  core/        Task / Agent / Grader interfaces + episode runner
  agents/      the single fixed ReAct scaffold (frozen across the study)
  adapters/    per-benchmark loaders + graders (cybergym implemented as stub)
  llm/         provider-agnostic client (model registry + training cutoffs)
  analysis/    transfer matrix, contamination gap, per-model scoring (RQ1–RQ3)
docs/          research proposal
tests/         estimator tests (run without API keys or benchmark data)
```

## Status

Harness skeleton + transfer-analysis code + tests are in place and run
offline. Next: vendor benchmark task data under `data/`, wire the provider
SDKs into `llm/client.py`, and implement the CyberGym / Cybench / CVE-Bench
graders. See the milestones in the proposal.

## Ethics & scope

Strictly defensive / evaluation work on existing public benchmark artifacts
(patched OSS-Fuzz vulns, CTF challenges, sandboxed CVE apps). Task environments
run without outbound network access. No new exploits against live systems.

## Dev

```bash
pip install -e ".[dev]"
pytest        # estimator tests run offline, no keys needed
```
