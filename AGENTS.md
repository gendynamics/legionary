# Agent workflow rules

1. **Planner** (Muse Spark 1.3): inspect each circuit with `count_gates` and choose strategies.
2. **Workers** (parallel): each applies one strategy (`level1`, `level2`, `level3`, `commute`,
   `opt1q`, `combo`) and returns a candidate circuit.
3. **Reviewer**: accept a candidate **only if `check_equiv` returns `ok: true`**; keep the
   smallest (total gates, then CX, then depth). If none pass, keep the original.
4. Report gate count, depth and CX count before and after.

Never pass `tool_choice` to `muse-spark-1.3` (only `"auto"` is accepted). Tests must not hit the network.
Entry point: `python -m quantum_optimizer.orchestrator --provider meta|baseten|none`.
