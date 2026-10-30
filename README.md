# quantum-optimizer

Agentic quantum circuit optimizer for the Meta Global AI Developer Hackathon.
A **Muse Spark 1.3** planner (Meta Responses API) picks strategies, a **Muse Code** swarm of
workers tries them in parallel, and a reviewer keeps only rewrites that pass an exact
unitary equivalence check (Qiskit `Operator.equiv`).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
pytest -q
python -m quantum_optimizer.orchestrator --provider none   # deterministic swarm, no API key
```

## Providers

| provider | base URL | env var | model |
|---|---|---|---|
| `meta` | https://api.meta.ai/v1 | `MODEL_API_KEY` | `muse-spark-1.3` |
| `baseten` | https://inference.baseten.co/v1 | `BASETEN_API_KEY` | `meta-models/muse-spark-open` |
| `none` | – | – | runs every strategy, no LLM |

`tool_choice` is never sent (Muse Spark 1.3 accepts only `"auto"`).

## Tools the planner calls

- `count_gates(qasm)` – total gates, depth, CX count, per-gate counts
- `check_equiv(before, after, tol)` – unitary equivalence (keep circuits ≲12 qubits)
- `simulate(qasm, shots, seed)` – Aer measurement counts

## Measured results (`--provider none`)

Benchmarks lowered to the `rz, sx, x, cx` basis (QFT 4 qubits, Grover 3 qubits, Bernstein–Vazirani secret 1011):

| circuit | strategy | gates before | gates after | depth before | depth after | cx before | cx after | equiv |
|---|---|---|---|---|---|---|---|---|
| qft4 | level2 | 48 | 42 | 28 | 25 | 18 | 18 | yes |
| grover3 | combo | 157 | 85 | 99 | 55 | 24 | 24 | yes |
| bv4 | level2 | 34 | 24 | 10 | 9 | 3 | 3 | yes |

## Layout

`quantum_optimizer/` (schemas, tools, planner, benchmarks, metrics, agents, reviewer, report,
orchestrator), `AGENTS.md`, `MUSE.md`, `.agents/skills/`, `tests/`.

## License

MIT © 2026 GenStro Intelligence, LLC

## Nemotron second-opinion reviewer (optional, Nebius)

Before each candidate rewrite goes through the exact `check_equiv` test, Legionary can send it to
NVIDIA **Nemotron 3 Super 120B** (`nvidia/nemotron-3-super-120b-a12b`) on Nebius Token Factory
(`https://api.tokenfactory.nebius.com/v1`, OpenAI-compatible chat completions) for an independent
review. It returns `verdict` (approve/reject/uncertain), `reasons` and `risk_notes`, logged next to
the equivalence result. The review is **advisory**: `check_equiv` still decides what is accepted,
and Muse Spark planning is unchanged.

- Enable: `export NEBIUS_API_KEY=...` (on by default when set; if unset each review is recorded as `skipped`).
- Disable: `--no-nemotron-review` or `LEGIONARY_NEMOTRON_REVIEW=0`.
- Errors or malformed replies are recorded as `error`; the run never crashes.
- Cost (estimate): ~1-2K tokens per review at $0.30/1M in, i.e. well under a cent per review.
