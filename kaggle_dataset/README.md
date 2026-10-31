# Quantum Circuit Optimization Agent Benchmark

Benchmark tasks for agents that optimize quantum circuits (and, as a secondary track, verified network routes).
An agent receives an input circuit and must return a smaller **equivalent** circuit. Each row records the **best known** result we measured.

Files: `circuits.csv` (26 rows, domain=quantum), `routing.csv` (13 rows, domain=routing), and `gen_dataset.py`, which regenerates both CSVs deterministically, offline, with no LLM calls (`/tmp/qo-venv/bin/python gen_dataset.py`; Qiskit 2.6.1, networkx 3.7).

## How best-known values were obtained (circuits)
Every circuit is lowered to the basis `rz, sx, x, cx` with `transpile(optimization_level=0)`, just as Legionary does (`quantum_optimizer.benchmarks._out`). Gate count, depth and CX count come from Legionary's `count_gates`. Candidates are:
- Qiskit `transpile` at optimization levels 0–3, `seed_transpiler=7`, with no coupling map
- Legionary offline chaining: `quantum_optimizer.rounds.optimize` using the deterministic all-strategies picker, 4 rounds, beam 3

Each candidate is checked against the original with `check_equiv` (Qiskit `Operator.equiv`, atol 1e-8). The winner is the lowest (gates, cx, depth) among the equivalent candidates. On a tie, the Qiskit level is credited first.

**`optimal_*` means BEST KNOWN from these measured runs, not a proven optimum.** None of these circuits uses a construction with a known optimum, such as QUEKO.

## Column dictionary: circuits.csv
| column | meaning |
|---|---|
| circuit_id | unique id |
| domain | always `quantum` |
| family | QFT / Grover / BV (Bernstein-Vazirani) / VQE ansatz / random |
| n_qubits | qubits in the circuit. BV uses secret length + 1 ancilla, so `bv4` has 5 qubits |
| n_gates, depth, cx_count | input circuit metrics after lowering to the basis |
| optimal_gates, optimal_depth, optimal_cx | metrics of the best known equivalent circuit |
| equivalence_proof | `operator_equiv_true` = the full unitary equivalence check passed. `not_checked` = skipped (only for >10 qubits; none in this release) |
| difficulty | small ≤4 qubits, medium 5–7, large ≥8 |
| source | the generator and its parameters/seed |
| provenance | the measured winner only, e.g. `qiskit-L2` or `legionary-chain:level1->level2 (offline, no LLM)` |
| send_order | number of passes in the winning sequence (1 for a Qiskit level, chain length for Legionary) |
| order_optimality | (best gate count over an exhaustive search of all 6 single and 36 two-step sequences of Legionary's strategies) ÷ (Legionary chaining's gate count). 1.0 means the chain search matched the exhaustive ≤2-step best. It is 1.0 for every row in this release |
| batch_size, superposition_gain, signature_scheme, qkd_links | **reserved, empty**: no measured data. Kept for future tasks |

## Column dictionary: routing.csv (synthetic)
We have no real routing data. These are **synthetic** seeded instances: `networkx.gnm_random_graph` with 10/20/50 nodes (25/60/150 edges), and seeds bumped until the graph is connected (the actual seed is in `source`). Edge cost is drawn from U[1,20] and latency from U[1,50] using `random.Random(seed)`. Source node is 0 and destination is n−1.
| column | meaning |
|---|---|
| route_id, domain, topology_size | id, `routing`, node count |
| attack_type | scenario label (hijack / route leak / spoofed announcement), assigned in rotation. It is a task definition, not simulated attack data |
| optimal_cost / optimal_latency | Dijkstra shortest-path length by cost / by latency (separate optimizations, so possibly different paths) |
| optimal_hops | minimum hop count (BFS) |
| signature_scheme | the signature scheme the scenario requires (ML-DSA / SLH-DSA, alternating). A task requirement, not a measurement |
| equivalence_proof | `path_cost_recomputed_true` = the path edge sums were recomputed, and the cost was cross-checked with Bellman-Ford |
| difficulty | small=10, medium=20, large=50 nodes |
| provenance | `dijkstra (networkx)`, with BFS for hops |
| qkd_links, send_order, order_optimality, batch_size, superposition_gain | **reserved, empty** |

## Scoring
score = Σᵢ w_d(i) · 1[beatsᵢ ∧ verifiedᵢ] / Σᵢ w_d(i) − λ · (total_tool_calls / N_entries)

- w_small = 1, w_medium = 2, w_large = 3
- λ = 0.01 (adjustable)
- **beats** = a strict improvement on `optimal_gates` for circuits, or on `optimal_cost` for routing (ties don't count)
- **verified** = the circuit passes operator equivalence with the input, or the submitted route is a valid src→dst path whose recomputed cost matches the claimed one and meets the scenario constraints

## Caveats
- The best known values are not proven optimal. Beating them is the point.
- Routing is synthetic, and Dijkstra is exactly optimal on it, so beating `optimal_cost` there is impossible on the base graph. That track is only meaningful once scenario constraints such as attack exclusions and signed hops are applied by the task harness.
- Reserved columns are intentionally empty.
- **Methods and strategies are intentionally excluded.** Provenance names only the winning source, not how to reproduce or improve it.

## License
Copyright © 2026 GenStro Intelligence, LLC. Suggested license: CC BY 4.0. **TODO: owner to confirm the license before publishing.**
