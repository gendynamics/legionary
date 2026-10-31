"""Regenerate the Quantum Circuit Optimization Agent Benchmark deterministically.
Run: /tmp/qo-venv/bin/python kaggle_dataset/gen_dataset.py  (from repo root or anywhere). Offline, no LLM."""
import csv, os, sys, itertools, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import QFT, TwoLocal, EfficientSU2
from qiskit.circuit.random import random_circuit
from quantum_optimizer.benchmarks import qft, grover, bernstein_vazirani, _out
from quantum_optimizer.tools import load, count_gates, check_equiv
from quantum_optimizer.agents import BASIS, STRATEGIES, work
from quantum_optimizer.rounds import optimize

EQ_MAX_QUBITS = 10
def diff(n): return "small" if n <= 4 else "medium" if n <= 7 else "large"
def vqe(lib, n, reps, seed):
    a = lib(n, reps=reps) if lib is EfficientSU2 else lib(n, "ry", "cx", reps=reps, entanglement="linear")
    rng = np.random.default_rng(seed)
    return _out(a.assign_parameters(rng.uniform(-np.pi, np.pi, a.num_parameters)).decompose())
def rnd(n, d, seed): return _out(random_circuit(n, d, max_operands=2, measure=False, seed=seed))

CIRCUITS = [  # id, family, n_qubits, qasm builder, source
 ("qft3","QFT",3,lambda: qft(3),"quantum_optimizer.benchmarks.qft n=3"),
 ("qft4","QFT",4,lambda: qft(4),"quantum_optimizer.benchmarks.qft n=4 (Legionary built-in)"),
 ("qft5","QFT",5,lambda: qft(5),"quantum_optimizer.benchmarks.qft n=5"),
 ("qft6","QFT",6,lambda: qft(6),"quantum_optimizer.benchmarks.qft n=6"),
 ("qftlib5","QFT",5,lambda: _out(QFT(5).decompose()),"generated: qiskit.circuit.library.QFT n=5"),
 ("qft8","QFT",8,lambda: qft(8),"quantum_optimizer.benchmarks.qft n=8"),
 ("grover2","Grover",2,lambda: grover(2),"quantum_optimizer.benchmarks.grover n=2 marked=3"),
 ("grover3","Grover",3,lambda: grover(3),"quantum_optimizer.benchmarks.grover n=3 marked=7 (Legionary built-in)"),
 ("grover3_m5","Grover",3,lambda: grover(3,5),"quantum_optimizer.benchmarks.grover n=3 marked=5"),
 ("grover4","Grover",4,lambda: grover(4),"quantum_optimizer.benchmarks.grover n=4 marked=15"),
 ("grover4_m6","Grover",4,lambda: grover(4,6),"quantum_optimizer.benchmarks.grover n=4 marked=6"),
 ("bv3","BV",4,lambda: bernstein_vazirani("101"),"quantum_optimizer.benchmarks.bernstein_vazirani secret=101"),
 ("bv4","BV",5,lambda: bernstein_vazirani("1011"),"quantum_optimizer.benchmarks.bernstein_vazirani secret=1011 (Legionary built-in)"),
 ("bv6","BV",7,lambda: bernstein_vazirani("110101"),"quantum_optimizer.benchmarks.bernstein_vazirani secret=110101"),
 ("bv8","BV",9,lambda: bernstein_vazirani("10110111"),"quantum_optimizer.benchmarks.bernstein_vazirani secret=10110111"),
 ("vqe_twolocal4_r2","VQE ansatz",4,lambda: vqe(TwoLocal,4,2,11),"generated: qiskit.circuit.library.TwoLocal(ry,cx,linear) n=4 reps=2 params seed=11"),
 ("vqe_twolocal6_r3","VQE ansatz",6,lambda: vqe(TwoLocal,6,3,12),"generated: qiskit.circuit.library.TwoLocal(ry,cx,linear) n=6 reps=3 params seed=12"),
 ("vqe_su2_3_r2","VQE ansatz",3,lambda: vqe(EfficientSU2,3,2,13),"generated: qiskit.circuit.library.EfficientSU2 n=3 reps=2 params seed=13"),
 ("vqe_su2_5_r2","VQE ansatz",5,lambda: vqe(EfficientSU2,5,2,14),"generated: qiskit.circuit.library.EfficientSU2 n=5 reps=2 params seed=14"),
 ("vqe_su2_8_r2","VQE ansatz",8,lambda: vqe(EfficientSU2,8,2,15),"generated: qiskit.circuit.library.EfficientSU2 n=8 reps=2 params seed=15"),
 ("rand3_d6_s1","random",3,lambda: rnd(3,6,1),"generated: qiskit random_circuit n=3 depth=6 max_operands=2 seed=1"),
 ("rand4_d8_s2","random",4,lambda: rnd(4,8,2),"generated: qiskit random_circuit n=4 depth=8 max_operands=2 seed=2"),
 ("rand5_d10_s3","random",5,lambda: rnd(5,10,3),"generated: qiskit random_circuit n=5 depth=10 max_operands=2 seed=3"),
 ("rand6_d10_s4","random",6,lambda: rnd(6,10,4),"generated: qiskit random_circuit n=6 depth=10 max_operands=2 seed=4"),
 ("rand8_d12_s5","random",8,lambda: rnd(8,12,5),"generated: qiskit random_circuit n=8 depth=12 max_operands=2 seed=5"),
 ("rand10_d10_s6","random",10,lambda: rnd(10,10,6),"generated: qiskit random_circuit n=10 depth=10 max_operands=2 seed=6"),
]
def key(m): return (m["total"], m["cx"], m["depth"])

def circuits():
    rows = []
    for cid, fam, n, build, src in CIRCUITS:
        q = build(); assert load(q).num_qubits == n, cid
        b = count_gates(q); check = n <= EQ_MAX_QUBITS
        cands = []
        for l in range(4):
            out = qasm2.dumps(transpile(load(q), basis_gates=BASIS, optimization_level=l, seed_transpiler=7))
            cands.append((f"qiskit-L{l}", out, 1))
        r = optimize(cid, q, rounds=4, beam=3)
        cands.append((f"legionary-chain:{r['strategy'].replace(' ', '')} (offline, no LLM)", r["qasm"], len(r["chain"])))
        scored = []
        for name, out, L in cands:
            m = count_gates(out)
            ok = check_equiv(q, out)["ok"] if check else None
            if ok is False: continue
            scored.append((key(m), name, m, L))
        scored.sort(key=lambda s: s[0])
        k, prov, m, L = scored[0]
        # exhaustive best over all 1- and 2-step strategy sequences (6 + 36)
        one = {s: work(s, q)["qasm"] for s in STRATEGIES}
        ex = [count_gates(v)["total"] for v in one.values()]
        for s1, s2 in itertools.product(STRATEGIES, STRATEGIES):
            ex.append(count_gates(work(s2, one[s1])["qasm"])["total"])
        lg = count_gates(r["qasm"])["total"]
        rows.append(dict(circuit_id=cid, domain="quantum", family=fam, n_qubits=n, n_gates=b["total"], depth=b["depth"],
            cx_count=b["cx"], optimal_gates=m["total"], optimal_depth=m["depth"], optimal_cx=m["cx"],
            equivalence_proof="operator_equiv_true" if check else "not_checked", difficulty=diff(n), source=src,
            provenance=prov, batch_size="", superposition_gain="", send_order=L,
            order_optimality=round(min(ex) / lg, 4) if lg else "", signature_scheme="", qkd_links=""))
        print(cid, b["total"], "->", m["total"], prov, flush=True)
    return rows

ATTACKS = ["hijack", "route leak", "spoofed announcement"]
def routing():
    import networkx as nx
    rows = []; i = 0
    for n, m in [(10, 25), (20, 60), (50, 150)]:
        for k in range(5 if n < 50 else 3):
            seed = 1000 + n * 10 + k
            G = nx.gnm_random_graph(n, m, seed=seed)
            while not nx.is_connected(G):
                seed += 1; G = nx.gnm_random_graph(n, m, seed=seed)
            rng = random.Random(seed)
            for u, v in G.edges:
                G[u][v]["cost"] = rng.randint(1, 20); G[u][v]["latency"] = rng.randint(1, 50)
            s, t = 0, n - 1
            pc = nx.dijkstra_path(G, s, t, weight="cost"); pl = nx.dijkstra_path(G, s, t, weight="latency")
            ph = nx.shortest_path(G, s, t)
            c = nx.dijkstra_path_length(G, s, t, weight="cost"); l = nx.dijkstra_path_length(G, s, t, weight="latency")
            ok = (sum(G[a][b]["cost"] for a, b in zip(pc, pc[1:])) == c and
                  sum(G[a][b]["latency"] for a, b in zip(pl, pl[1:])) == l and
                  c == nx.bellman_ford_path_length(G, s, t, weight="cost"))
            rows.append(dict(route_id=f"rt{n}_{k}", domain="routing", topology_size=n, attack_type=ATTACKS[i % 3],
                optimal_cost=c, optimal_latency=l, optimal_hops=len(ph) - 1, signature_scheme=["ML-DSA", "SLH-DSA"][i % 2],
                qkd_links="", send_order="", order_optimality="",
                equivalence_proof="path_cost_recomputed_true" if ok else "path_cost_mismatch",
                difficulty={10: "small", 20: "medium", 50: "large"}[n],
                source=f"synthetic: networkx gnm_random_graph n={n} m={m} seed={seed}; edge cost~U[1,20], latency~U[1,50] via random.Random(seed); src=0 dst={n-1}",
                provenance="dijkstra (networkx); hops via BFS; cost cross-checked with bellman_ford", batch_size="", superposition_gain=""))
            i += 1
    return rows

def write(name, rows):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    write("routing.csv", routing()); write("circuits.csv", circuits())
