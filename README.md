# legionary

Agentic quantum circuit optimizer for the Meta Global AI Developer Hackathon.
A **Muse Spark 1.3** planner (Meta Responses API) picks strategies, a **Muse Code** swarm of
workers tries them in parallel, and a reviewer keeps only rewrites that pass an exact
unitary equivalence check (Qiskit `Operator.equiv`).

## Quick start

Calling the venv's Python by path (`.venv/bin/python3`) makes every command use the venv,
even if `python` or `pytest` on your PATH points at a system install.

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -e ".[test]"   # installs the package + all deps + pytest
.venv/bin/python3 -m pytest -q                  # 69 tests, offline (no keys, no network)
.venv/bin/python3 -m quantum_optimizer.orchestrator --provider none   # deterministic swarm, no API key
```

Type each command on its own line; pasting several at once can add hidden characters.

## Live 3D viewer

Run the optimizer and open the live 3D window together (no API key needed):

```bash
.venv/bin/python3 -m quantum_optimizer.orchestrator --only grover3 --viewer --angle 40 --director-backend off
```

Full agent run on Fireworks (Muse Spark open weights) with the viewer:

```bash
export FIREWORKS_API_KEY=...
.venv/bin/python3 -m quantum_optimizer.orchestrator --planner agent --backend fireworks --only grover3 --viewer --angle 40
```

Viewer only, with a synthetic demo run (not real data):

```bash
.venv/bin/python3 -m quantum_optimizer.viewer --demo --director-backend off
```

Controls: drag to orbit, scroll to zoom, Esc to close. On macOS the window can open behind
your editor; click the Python icon in the Dock. "waiting for stream" means no run is feeding
the window yet: launch it through the orchestrator with `--viewer` instead of on its own.

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

## Measured results (`--provider none`, default `--rounds 4 --beam 3`)

Benchmarks lowered to the `rz, sx, x, cx` basis (QFT 4 qubits, Grover 3 qubits, Bernstein–Vazirani secret 1011):

| circuit | strategy | gates before | gates after | depth before | depth after | cx before | cx after | equiv |
|---|---|---|---|---|---|---|---|---|
| qft4 | level2 | 48 | 42 | 28 | 25 | 18 | 18 | yes |
| grover3 | level1 -> level2 | 157 | 81 | 99 | 55 | 24 | 24 | yes |
| bv4 | level2 | 34 | 24 | 10 | 9 | 3 | 3 | yes |

**Multi-round chaining.** The optimizer runs up to `--rounds N` (default 4) rounds. Each round applies
strategies to the best `--beam K` circuits so far (default 3; 1 = greedy), and every candidate is
equivalence-checked against the **original** circuit. It stops at a fixed point. With `--provider meta|baseten`,
Muse picks and orders strategies each round, given the chain so far. Offline, it tries all of them.
The discovered chain is shown in the `strategy` column. Greedy (`--beam 1`) stalls at grover3 `combo` = 85.
The beam finds `level1 -> level2` = 81, below Qiskit's best single preset level (level 1: 93 gates, depth 55).
qft4 and bv4 tie Qiskit levels 2/3. CX counts are unchanged on all three.

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
