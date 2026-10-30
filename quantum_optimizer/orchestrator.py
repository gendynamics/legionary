"""CLI: python -m quantum_optimizer.orchestrator [--provider meta|baseten|none] [--[no-]nemotron-review]"""
import argparse, logging, re
from .agents import STRATEGIES, swarm
from .benchmarks import all_benchmarks
from .reviewer import review
from . import nemotron
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
    p.add_argument("--nemotron-review", dest="nemotron", action=argparse.BooleanOptionalAction, default=None,
                   help="independent Nemotron review via Nebius (default: on; skipped if NEBIUS_API_KEY unset)")
    a = p.parse_args(argv)
    lg = logging.getLogger("legionary")
    if not lg.handlers:
        h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    use = nemotron.enabled(a.nemotron)
    second = nemotron.review_rewrite if use else None
    rows, reviews = [], []
    for name, qasm in all_benchmarks().items():
        recs = []
        best = review(qasm, swarm(qasm, pick(name, qasm, a.provider)), second_opinion=second, records=recs)
        rows.append((name, qasm, best))
        reviews += [(name, r) for r in recs]
    print(table(rows))
    if use:
        print(review_table(reviews))


def review_table(reviews):
    out = ["", "| circuit | rewrite | nemotron status | nemotron verdict | equiv |", "|---|---|---|---|---|"]
    for name, r in reviews:
        n = r.get("nemotron", {})
        out.append(f"| {name} | {r['strategy']} | {n.get('status')} | {n.get('verdict') or '-'} | "
                   f"{'yes' if r['equiv'] else 'NO'} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
