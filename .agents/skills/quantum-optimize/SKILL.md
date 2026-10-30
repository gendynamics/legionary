---
name: quantum-optimize
description: Run the full planner -> parallel workers -> reviewer optimization on the benchmarks.
---
Run `python -m quantum_optimizer.orchestrator --provider meta` (or `baseten`, or `none`
for the deterministic swarm only) and show the resulting markdown table.
Run it as a workflow so the workers are visible in `/workflows`.
