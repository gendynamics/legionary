"""The real tools agents call."""
from qiskit import qasm2, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


def load(qasm):
    # Legacy instructions let qelib1 gates Qiskit emits (sx, cp, ...) parse back.
    return qasm2.loads(qasm, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def count_gates(qasm):
    qc = load(qasm)
    ops = dict(qc.count_ops())
    return {"total": qc.size(), "depth": qc.depth(),
            "cx": ops.get("cx", 0), "ops": ops}


def check_equiv(before, after, tol=1e-8):
    a, b = load(before), load(after)
    a.remove_final_measurements()
    b.remove_final_measurements()
    ok = Operator(a).equiv(Operator(b), atol=tol)
    return {"ok": bool(ok)}


def simulate(qasm, shots=1024, seed=7):
    qc = load(qasm)
    if not qc.cregs:
        qc.measure_all()
    sim = AerSimulator(seed_simulator=seed)
    res = sim.run(transpile(qc, sim), shots=shots).result()
    return {"counts": res.get_counts()}
