"""CLI: python -m quantum_optimizer.orchestrator [--provider meta|baseten|none] [--rounds N] [--beam K] [--only NAME ...] [--[no-]nemotron-review]"""
import argparse, logging, re
from .agents import STRATEGIES, swarm
from .benchmarks import all_benchmarks
from .reviewer import review
from . import nemotron
from .report import table
from .planner import plan
from .rounds import optimize


def pick(name, qasm, provider, ai=None, round_no=1, chain=None):
    if provider == "none":
        return list(STRATEGIES)
    prior = (f"Round {round_no}. Strategies already applied in order: {' -> '.join(chain)}. "
             f"Pick what to try NEXT on this already-optimized circuit, best first. ") if chain else ""
    text = plan(f"Circuit '{name}'. {prior}Use count_gates to inspect it, then reply with a "
                f"comma-separated list of strategies to try from: {', '.join(STRATEGIES)}.\n"
                f"QASM:\n{qasm}", where=provider, ai=ai) or ""
    hits = [(m.start(), s) for s in STRATEGIES for m in [re.search(rf"\b{s}\b", text)] if m]
    chosen = [s for _, s in sorted(hits)]  # keep the planner's order
    return chosen or list(STRATEGIES)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["meta", "baseten", "none"], default="none")
    p.add_argument("--nemotron-review", dest="nemotron", action=argparse.BooleanOptionalAction, default=None,
                   help="independent Nemotron review via Nebius (default: on; skipped if NEBIUS_API_KEY unset)")
    p.add_argument("--rounds", type=int, default=4, help="max optimize rounds (stops early at a fixed point)")
    p.add_argument("--beam", type=int, default=3, help="circuits kept per round (1 = greedy)")
    p.add_argument("--only", nargs="*", help="benchmark names to run (default: all)")
    a = p.parse_args(argv)
    lg = logging.getLogger("legionary")
    if not lg.handlers:
        h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    use = nemotron.enabled(a.nemotron)
    second = nemotron.review_rewrite if use else None
    rows, reviews = [], []
    picker = None if a.provider == "none" else (lambda n, q, r, c: pick(n, q, a.provider, round_no=r, chain=c))
    for name, qasm in all_benchmarks().items():
        if a.only and name not in a.only:
            continue
        recs = []
        best = optimize(name, qasm, picker=picker, rounds=a.rounds, beam=a.beam, second_opinion=second, records=recs)
        rows.append((name, qasm, best))
        reviews += [(name, r) for r in recs]
    print(table(rows))
    if use:
        print(review_table(reviews))
    for name, _, best in rows:
        print(f"chain {name}: {best['strategy']} ({best['rounds_run']} rounds)")


def review_table(reviews):
    out = ["", "| circuit | rewrite | nemotron status | nemotron verdict | equiv |", "|---|---|---|---|---|"]
    for name, r in reviews:
        n = r.get("nemotron", {})
        out.append(f"| {name} | {r['strategy']} | {n.get('status')} | {n.get('verdict') or '-'} | "
                   f"{'yes' if r['equiv'] else 'NO'} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
