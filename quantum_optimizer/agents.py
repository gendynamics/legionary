"""Swarm workers: each tries a different deterministic strategy, in parallel."""
from concurrent.futures import ThreadPoolExecutor
from qiskit import qasm2, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (CommutativeCancellation, Optimize1qGatesDecomposition,
                                      InverseCancellation)
from qiskit.circuit.library import CXGate, HGate, XGate
from .tools import load

BASIS = ["rz", "sx", "x", "cx"]


def _lower(qc):
    return transpile(qc, basis_gates=BASIS, optimization_level=0)


def s_level(level):
    return lambda qc: transpile(qc, basis_gates=BASIS, optimization_level=level, seed_transpiler=7)


def s_commute(qc):
    return PassManager([CommutativeCancellation(basis_gates=BASIS)]).run(_lower(qc))


def s_1q(qc):
    return PassManager([Optimize1qGatesDecomposition(basis=BASIS)]).run(_lower(qc))


def s_combo(qc):
    pm = PassManager([InverseCancellation([CXGate(), HGate(), XGate()]),
                      CommutativeCancellation(basis_gates=BASIS),
                      Optimize1qGatesDecomposition(basis=BASIS)])
    return pm.run(pm.run(_lower(qc)))


STRATEGIES = {"level1": s_level(1), "level2": s_level(2), "level3": s_level(3),
              "commute": s_commute, "opt1q": s_1q, "combo": s_combo}


def work(name, qasm):
    try:
        return {"strategy": name, "qasm": qasm2.dumps(STRATEGIES[name](load(qasm)))}
    except Exception as e:
        return {"strategy": name, "error": str(e)}


def swarm(qasm, names=None, workers=6):
    names = [n for n in (names or STRATEGIES) if n in STRATEGIES] or list(STRATEGIES)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda n: work(n, qasm), names))
