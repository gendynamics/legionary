"""Markdown results table."""
from .tools import count_gates, check_equiv


def table(rows):
    out = ["| circuit | strategy | gates before | gates after | depth before | depth after | cx before | cx after | equiv |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, before, best in rows:
        b, a = count_gates(before), count_gates(best["qasm"])
        eq = check_equiv(before, best["qasm"])["ok"]
        out.append(f"| {name} | {best['strategy']} | {b['total']} | {a['total']} | {b['depth']} | "
                   f"{a['depth']} | {b['cx']} | {a['cx']} | {'yes' if eq else 'NO'} |")
    return "\n".join(out)
