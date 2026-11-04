# Legionary

**An AI agent that rewrites quantum circuits and proves it didn't break them.**

## The story
On today's quantum hardware, every gate adds noise, so fewer gates mean better answers. Squeezing circuits down is still slow, manual expert work, and a single wrong rewrite silently changes the answer. Legionary hands that job to Meta's Muse Spark as an agent. The model searches for shorter circuits, and math checks every result it proposes. Nothing ships unless it is provably the same computation as the original.

## How it works
**Agent loop.** Muse Spark runs as a tool-calling agent with a hard budget of 50 tool calls across up to 60 turns. When 5 calls are left, it is told to wrap up. Every tool result goes back to the model as structured JSON, including errors such as an unknown circuit ID or strategy, so a bad call costs one step and never crashes the run.

**Tool design.** The agent has five tools: `list_strategies`, `run_pass`, `metrics`, `check_equivalence` and `finish`. Circuits never pass through the model as text. They live in a server-side store, where each `run_pass` creates a new circuit ID (`c0`, `c1`, …) that records its parent's lineage. The agent can branch from any earlier result instead of only the latest one. This keeps the model's context small and rules out hand-edited, malformed circuits.

**Search structure.** Underneath the agent sits a deterministic engine that runs a beam search, which keeps the few best candidates each round instead of only one. Each round runs the candidate passes in parallel on a 6-worker pool against every circuit in the beam. The 3 best distinct equivalent circuits survive, and duplicates are dropped by exact match. Candidates are ranked by gate count first, then two-qubit (CX) count, then depth. The search stops at a fixed point, when a round brings no improvement, or after 4 rounds. A beam wider than one matters on Grover: picking only the single best step each round stalls at 85 gates, while keeping 3 candidates reaches 81.

**Verification gate.** Every candidate is checked against the **original** input circuit, never just its parent, so errors can't pile up across a chain of rewrites. The check strips final measurements and compares the full unitary matrices with Qiskit's `Operator.equiv` at tolerance 1e-8, allowing only a difference in global phase. `finish` is refused unless the chosen circuit passes. After the loop, the result is checked again in code. If the model never finished validly, Legionary returns the smallest verified circuit it saw, or the original circuit.

**Planner and reviewer.** The planner proposes and the reviewer decides. The reviewer accepts only equivalent circuits and keeps the best by gates, then CX, then depth. An optional second-opinion review by NVIDIA Nemotron can flag risky rewrites, but it is advisory only. Only the math check can accept a circuit.

**Cross-checked results.** The deterministic beam search serves as an independent verifier: run separately with no model in the loop, it confirms the agent's 81-gate answer, on top of the equivalence gate that already proved it exact. Two independent runs, on Fireworks and on Nebius, landed on the same 81 gates. The test suite (69 tests) mocks every model call and checks the gate, the fallback and the search.

**Live 3D viewer.** A native, cinematic 3D scene streams the agent's run in real time: gates as glowing diamond frames, removed gates left behind as still frames, live gate, depth and CX counters, and the equivalence badge.

*(The pass library itself, meaning which rewrites exist and how they are tuned and combined, is proprietary and intentionally not described here.)*

## Meta tooling
- **Muse Spark as the planner.** It runs through the Responses API with tool calling on the Meta Model API (`muse-spark-1.3`), and also runs on Muse Spark Open through Meta's inference partners Fireworks and Nebius. The same agent code works on all three.
- **Muse Spark as the visual director.** At startup, Muse Spark chooses the live viewer's palette, flame color and look of removed gates from reference images. The choice is validated, cached, and backed by a built-in default.
- **Built on Meta's open-weights ecosystem.** It is backend-portable, with automatic fallback across providers.

**Meta tooling evidence.** The planner is Muse Spark driving five tools (`list_strategies`, `run_pass`, `metrics`, `check_equivalence`, `finish`) through tool calls. On the Meta Model API (`muse-spark-1.3` at api.meta.ai), the agent uses Meta's native Responses API (`responses.create` with tool definitions). On Muse Spark Open via partners, the same tools run over the OpenAI-compatible Chat Completions path: Fireworks is pinned to Chat Completions, and Nebius tries the Responses API first, then falls back to Chat Completions. The measured Grover runs below were on Muse Spark Open via Fireworks and Nebius.

## Results (measured Oct 31, 2026)
**3-qubit Grover search, compiled to the hardware gate set**

| | Before | After | Change |
|---|---|---|---|
| Gates | 157 | **81** | 48% fewer |
| Depth | 99 | **55** | 44% shallower |
| CX (two-qubit) | 24 | 24 | unchanged |
| Equivalence | | **PASS** | exact unitary match |

- The agent finished on its own after 47 of 50 allowed tool calls.
- The run used 45,721 input and 6,190 output tokens on Muse Spark Open via Fireworks, costing **about $0.08**.
- A second run on Nebius reached the same 81 gates for about $0.07.

### Baseline: Qiskit transpiler
Same Grover circuit, same gate set (rz, sx, x, cx), no coupling map, counted and equivalence-checked with Legionary's own tools. Qiskit 2.6.1, `seed_transpiler=42`; level 3 also tried with seeds 0–19 (best shown).

| | Gates | Depth | CX | Equivalence |
|---|---|---|---|---|
| Original | 157 | 99 | 24 | |
| Qiskit `optimization_level=1` (best Qiskit result) | 93 | 55 | 24 | PASS |
| Qiskit `optimization_level=2` | 100 | 64 | 24 | PASS |
| Qiskit `optimization_level=3` (best of 20 seeds) | 100 | 64 | 24 | PASS |
| **Legionary** | **81** | **55** | 24 | PASS |

Legionary uses 12 fewer gates than Qiskit's best setting (81 vs 93, about 13% fewer) at the same depth and CX count.

## Demo
The 40-second video shows a full live replay of the Grover run in the 3D viewer:
- The original 157-gate circuit.
- The agent's passes ticking by as gates collapse to 81.
- A camera flight through the diamond-framed gates.
- A finish on the green equivalence PASS with the final numbers.

## What's next
- More benchmarks: QFT, adders, Toffoli-heavy circuits and QAOA.
- A simulation-based check for circuits too large for a full unitary check.
- Parallel planner agents with an independent reviewer agent.
