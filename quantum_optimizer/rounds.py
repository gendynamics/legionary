"""Multi-round strategy chaining (small beam search).

Each round runs the chosen strategies on each circuit in the beam (the current best few
circuits). Every candidate goes through the reviewer, which checks equivalence against
the ORIGINAL circuit (check_equiv stays the sole authority; Nemotron stays advisory).
The top `beam` distinct equivalent circuits by the reviewer objective (gates, cx, depth)
survive. The loop stops when a round brings no improvement over the best so far
(fixed point) or after max rounds. A beam > 1 matters: greedy picks grover3's best
single step (combo, 85) and stalls, while level1 (93) -> level2 reaches 81.
"""
import logging
from .agents import STRATEGIES, swarm
from .reviewer import review
from .tools import count_gates

log = logging.getLogger("legionary.rounds")


def _key(m):
    return (m["total"], m["cx"], m["depth"])


def optimize(name, original, picker=None, rounds=4, beam=3, second_opinion=None, records=None,
             swarm_fn=swarm, review_fn=review):
    """picker(name, qasm, round_no, chain) -> ordered strategy names (None/empty = all, deterministic).
    Returns the best dict plus: chain (list of strategies), rounds_run, history."""
    start = {"strategy": "original", "qasm": original, "equiv": True, **count_gates(original), "chain": []}
    best, frontier, history = start, [start], []
    seen = {original}
    for r in range(1, max(1, rounds) + 1):
        pool = []
        for node in frontier:
            names = (picker(name, node["qasm"], r, list(node["chain"])) if picker else None) or list(STRATEGIES)
            for c in swarm_fn(node["qasm"], names):
                recs = []
                got = review_fn(node["qasm"], [c], second_opinion=second_opinion, records=recs,
                                reference=original)
                if records is not None:
                    records += [dict(x, round=r, chain=" -> ".join(node["chain"] + [c["strategy"]]))
                                for x in recs]
                if got["strategy"] == "original" or got["qasm"] in seen:
                    continue  # rejected by reviewer, no change, or duplicate
                seen.add(got["qasm"])
                pool.append({**got, "chain": node["chain"] + [got["strategy"]]})
        pool.sort(key=_key)
        top = pool[0] if pool else None
        improved = top is not None and _key(top) < _key(best)
        history.append({"round": r, "candidates": len(pool),
                        "best_chain": " -> ".join(top["chain"]) if top else None,
                        **({k: top[k] for k in ("total", "cx", "depth")} if top else {})})
        if not improved:
            log.info("%s round %d: no improvement, fixed point at %s", name, r,
                     " -> ".join(best["chain"]) or "original")
            break
        best = top
        frontier = pool[:max(1, beam)]
        log.info("%s round %d: gates=%d cx=%d depth=%d chain=%s", name, r, best["total"], best["cx"],
                 best["depth"], " -> ".join(best["chain"]))
    return {**best, "strategy": " -> ".join(best["chain"]) or "original",
            "rounds_run": len(history), "history": history}
