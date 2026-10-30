---
name: planner
description: Plan and run quantum circuit optimization.
---
To optimize a circuit, run one of these:

- Meta endpoint:    python -m quantum_optimizer.planner meta  bench/qft.qasm
- Baseten endpoint: python -m quantum_optimizer.planner baseten bench/qft.qasm

Use meta by default. Use baseten to get a second opinion.
Only accept a result if check_equiv returned ok true.
Report gate count and depth before and after.
