"""Benchmark circuits (QFT, Grover, Bernstein-Vazirani) as QASM 2 strings."""
import math
from qiskit import QuantumCircuit, qasm2

def _out(qc):
    # Lower to the same basis the workers use (rz, sx, x, cx) so counts compare fairly.
    from qiskit import transpile
    return qasm2.dumps(transpile(qc, basis_gates=["rz", "sx", "x", "cx"],
                                 optimization_level=0))


def qft(n=4):
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(math.pi / 2 ** (k - j), k, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return _out(qc)


def grover(n=3, marked=None):
    marked = (1 << n) - 1 if marked is None else marked
    qc = QuantumCircuit(n)
    qc.h(range(n))

    def flip_marked():
        for i in range(n):
            if not (marked >> i) & 1:
                qc.x(i)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for i in range(n):
            if not (marked >> i) & 1:
                qc.x(i)

    iters = max(1, int(math.floor(math.pi / 4 * math.sqrt(2 ** n))))
    for _ in range(iters):
        flip_marked()
        qc.h(range(n)); qc.x(range(n))
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(range(n)); qc.h(range(n))
    return _out(qc)


def bernstein_vazirani(secret="1011"):
    n = len(secret)
    qc = QuantumCircuit(n + 1)
    qc.x(n); qc.h(range(n + 1))
    for i, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc.cx(i, n)
    qc.h(range(n + 1))
    return _out(qc)


def all_benchmarks():
    return {"qft4": qft(4), "grover3": grover(3), "bv4": bernstein_vazirani("1011")}
