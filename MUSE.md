# Quantum Optimizer with Muse Code

This project uses Meta's Muse stack for the Meta Global AI Developer Hackathon:

- **Muse Spark 1.3** (Meta Responses API, `https://api.meta.ai/v1`) is the planner. It calls
  the real tools in `quantum_optimizer/tools.py` (`count_gates`, `check_equiv`, `simulate`)
  and picks which optimization strategies the swarm should try.
- **Muse Code** runs the workflow: ask it to "use a workflow" with the `quantum-optimize`
  skill, and watch the planner, parallel workers and reviewer in `/workflows`.
- Project rules live in `AGENTS.md`; skills live in `.agents/skills/<name>/SKILL.md`.
  Run `muse skills validate` to check them.
