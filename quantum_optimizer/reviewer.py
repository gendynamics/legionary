"""Reviewer: picks the smallest candidate that passes check_equiv."""
from .tools import check_equiv, count_gates


def review(before, candidates):
    base = count_gates(before)
    best = {"strategy": "original", "qasm": before, "equiv": True, **base}
    for c in candidates:
        if "error" in c:
            continue
        if not check_equiv(before, c["qasm"])["ok"]:
            continue
        m = count_gates(c["qasm"])
        if (m["total"], m["cx"], m["depth"]) < (best["total"], best["cx"], best["depth"]):
            best = {"strategy": c["strategy"], "qasm": c["qasm"], "equiv": True, **m}
    return best
