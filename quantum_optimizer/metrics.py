"""Before/after deltas."""
from .tools import count_gates


def deltas(before, after):
    b, a = count_gates(before), count_gates(after)
    return {k: {"before": b[k], "after": a[k], "delta": a[k] - b[k]}
            for k in ("total", "depth", "cx")}
