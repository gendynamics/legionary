# Legionary

**A Muse Spark agent that shrinks quantum circuits and proves every rewrite is exact.**

Theme: Push the Frontier · Track: Indie developer

---

## 1. Impact (30%)

**The problem.** On today's quantum hardware, every gate adds noise, so a shorter circuit gives a better answer on the same machine. Shrinking circuits is still slow, manual expert work, and one wrong rewrite silently changes what the circuit computes. You often won't find out until the results are already wrong.

**What Legionary changes.** Legionary hands circuit shrinking to Muse Spark running as an agent, and it ships nothing that math hasn't proven equivalent to the original. The result is a smaller circuit you can trust, produced in minutes for cents.

**Measured result (Oct 31, 2026), 3-qubit Grover search compiled to the hardware gate set:**

| | Before | After | Change |
|---|---|---|---|
| Gates | 157 | **81** | 48% fewer |
| Depth | 99 | **55** | 44% shallower |
| CX (two-qubit) | 24 | 24 | unchanged |
| Equivalence | | **PASS** | exact unitary match |

**Compared with the industry-standard tool.** We ran the Qiskit 2.6.1 transpiler on the same circuit with the same gate set (rz, sx, x, cx) and no coupling map. Gates were counted and checked with the same tools.

| | Gates | Depth | CX | Equivalence |
|---|---|---|---|---|
| Original | 157 | 99 | 24 | |
| Qiskit `optimization_level=1` (best Qiskit) | 93 | 55 | 24 | PASS |
| Qiskit `optimization_level=2` | 100 | 64 | 24 | PASS |
| Qiskit `optimization_level=3` (best of 20 seeds) | 100 | 64 | 24 | PASS |
| **Legionary** | **81** | **55** | 24 | PASS |

Legionary uses **12 fewer gates than Qiskit's best setting** (81 vs 93, about 13% fewer) at the same depth and CX count. A full agent run cost **about $0.08**.

**Who it's for.** It's for researchers and developers who run circuits on noisy hardware and want shorter circuits without hand-tuning, and without risking a broken computation.

---

## 2. Use of Meta AI tooling (25%)

Muse Spark is load-bearing. It is the planner that decides every optimization step. Without it there is no agent, only a fixed pipeline.

- **Muse Spark as the planner.** Muse Spark drives five tools through native tool calling: `list_strategies`, `run_pass`, `metrics`, `check_equivalence` and `finish`. It picks which rewrite to try next, which earlier circuit to branch from, and when to stop.
- **Muse Spark Open (open weights).** The measured Grover runs used Muse Spark Open through Meta's inference partners Fireworks and Nebius. Two independent runs, one on each provider, reached the same 81 gates (about $0.08 and $0.07). This is the open-weight stage for the Best Build on Meta Open Weight Models award.
- **Meta Model API path.** The same agent code targets `muse-spark-1.3` at api.meta.ai through Meta's Responses API (`responses.create` with tool definitions). One codebase runs on all three backends, with automatic fallback: Nebius tries the Responses API first and falls back to Chat Completions, while Fireworks always uses Chat Completions.
- **Muse Spark as the visual director.** At startup, Muse Spark looks at reference images and chooses the live viewer's palette, flame color and the look of removed gates. Its choice is validated, cached, and backed by a built-in default.
- **Efficient by design.** Circuits never pass through the model as text. They live in a server-side store under IDs (`c0`, `c1`, …), so the model reasons about strategy rather than parsing gate lists. The full run took 45,721 input and 6,190 output tokens.

---

## 3. Technical depth and execution (20%)

**Agent loop.** The agent has a hard budget of 50 tool calls across up to 60 turns, and it is told to wrap up when 5 calls are left. It finished on its own after **47 of 50 calls**. Every tool result goes back as structured JSON, including errors such as an unknown circuit ID or strategy, so a bad call costs one step and never crashes the run.

**Lineage-tracked circuit store.** Each `run_pass` creates a new circuit ID that records its parent. The agent can branch from any earlier result, not just the latest one, and it can't hand-edit or corrupt a circuit.

**Beam search engine.** Underneath the agent sits a deterministic beam search:
- Each round runs candidate passes in parallel on a 6-worker pool.
- The 3 best distinct equivalent circuits survive each round, and exact duplicates are dropped.
- Candidates are ranked by gates, then CX, then depth.
- The search stops when a round brings no improvement, or after 4 rounds.

The beam width matters. A greedy search that keeps only the single best step stalls at 85 gates, while a beam of 3 reaches 81.

**Verification gate, the core guarantee.**
- Every candidate is checked against the **original** circuit, never just its parent, so errors can't pile up across a chain of rewrites.
- The check strips final measurements and compares full unitaries with Qiskit's `Operator.equiv` at a tolerance of 1e-8, allowing only a global-phase difference.
- `finish` is refused unless the chosen circuit passes the check.
- After the loop, the result is checked again in code. If the model never finished validly, Legionary returns the smallest verified circuit it saw, or else the original. The model can't ship a wrong answer.

**Planner and reviewer.** The planner proposes, and a reviewer accepts only equivalent circuits, keeping the best by gates, then CX, then depth.

**Independently cross-checked.** The deterministic beam search, run alone with no model in the loop, confirms the agent's 81-gate answer. Two runs on different providers agree.

**Tested.** 69 tests mock every model call and cover the verification gate, the fallback path and the search.

*(The pass library itself, meaning which rewrites exist and how they are tuned and combined, is proprietary and intentionally not described here.)*

---

## 4. Demo (25%)

The video is a live replay of the Grover run in Legionary's native, real-time 3D viewer:
- It starts from the original 157-gate circuit.
- The agent's passes tick by as gates collapse to 81, with removed gates left behind as still frames.
- Live counters track gates, depth and CX.
- A camera flight moves through the diamond-framed gates.
- It finishes on the green equivalence **PASS** with the final numbers.

**Run it yourself:**
```
python -m quantum_optimizer.orchestrator --planner agent --backend fireworks --only grover3
```
Set `FIREWORKS_API_KEY`, `NEBIUS_API_KEY`, or `META_API_KEY` for the backend you choose. The deterministic engine runs with no API key at all.

---

## What's next
- More benchmarks: QFT, adders, Toffoli-heavy circuits and QAOA.
- A simulation-based check for circuits too large for a full unitary check.
- Parallel Muse Spark planner agents with an independent reviewer agent.
