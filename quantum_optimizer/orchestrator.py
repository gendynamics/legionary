"""CLI: python -m quantum_optimizer.orchestrator [--provider meta|baseten|none]"""
import argparse, re
from .agents import STRATEGIES, swarm
from .benchmarks import all_benchmarks
from .reviewer import review
from .report import table
from .planner import plan


def pick(name, qasm, provider, ai=None):
    if provider == "none":
        return list(STRATEGIES)
    text = plan(f"Circuit '{name}'. Use count_gates to inspect it, then reply with a "
                f"comma-separated list of strategies to try from: {', '.join(STRATEGIES)}.\n"
                f"QASM:\n{qasm}", where=provider, ai=ai) or ""
    chosen = [s for s in STRATEGIES if re.search(rf"\b{s}\b", text)]
    return chosen or list(STRATEGIES)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["meta", "baseten", "none"], default="none")
    a = p.parse_args(argv)
    rows = []
    for name, qasm in all_benchmarks().items():
        best = review(qasm, swarm(qasm, pick(name, qasm, a.provider)))
        rows.append((name, qasm, best))
    print(table(rows))


if __name__ == "__main__":
    main()
