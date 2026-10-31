"""Reviewer: picks the smallest candidate that passes check_equiv.

Optional: an independent Nemotron review (advisory) runs on each rewrite BEFORE check_equiv.
check_equiv remains the sole authority for acceptance.
"""
import logging
from .tools import check_equiv, count_gates

log = logging.getLogger("legionary.reviewer")


def review(before, candidates, second_opinion=None, records=None, reference=None):
    """second_opinion: callable(before, after, strategy) -> dict, or None to skip.
    records: optional list; one record per candidate is appended with both results.
    reference: circuit equivalence is checked against (default: before). The multi-round
    loop passes the ORIGINAL circuit so every round is checked against the true input."""
    ref = before if reference is None else reference
    base = count_gates(before)
    best = {"strategy": "original", "qasm": before, "equiv": True, **base}
    for c in candidates:
        if "error" in c:
            continue
        rec = {"strategy": c["strategy"]}
        if second_opinion is not None:
            rec["nemotron"] = second_opinion(before, c["qasm"], c["strategy"])
        ok = check_equiv(ref, c["qasm"])["ok"]
        rec["equiv"] = ok
        if records is not None:
            records.append(rec)
        if "nemotron" in rec:
            n = rec["nemotron"]
            log.info("rewrite %s: nemotron=%s/%s equiv=%s", c["strategy"], n.get("status"), n.get("verdict"), ok)
        if not ok:
            continue
        m = count_gates(c["qasm"])
        if (m["total"], m["cx"], m["depth"]) < (best["total"], best["cx"], best["depth"]):
            best = {"strategy": c["strategy"], "qasm": c["qasm"], "equiv": True, **m}
    return best
