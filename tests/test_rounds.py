"""Offline tests for multi-round chaining (no API calls)."""
from quantum_optimizer import tools
from quantum_optimizer.benchmarks import all_benchmarks
from quantum_optimizer.rounds import optimize
from quantum_optimizer import orchestrator

H = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
PAD3 = H + "x q[0];\nx q[0];\nx q[0];\nx q[0];\nh q[0];\n"   # equivalent to H + "h q[0];"
PAD1 = H + "x q[0];\nx q[0];\nh q[0];\n"
GOAL = H + "h q[0];\n"
BAD = H + "z q[0];\n"                                      # smaller but NOT equivalent


def fake_swarm(table):
    def f(qasm, names):
        return [{"strategy": "peel", "qasm": table[qasm]}] if qasm in table else []
    return f


def test_stops_at_fixed_point_and_records_chain():
    best = optimize("t", PAD3, rounds=10, swarm_fn=fake_swarm({PAD3: PAD1, PAD1: GOAL, GOAL: GOAL}))
    assert best["qasm"] == GOAL and best["chain"] == ["peel", "peel"]
    assert best["strategy"] == "peel -> peel"
    assert best["rounds_run"] == 3          # 2 improving rounds + 1 that finds nothing new
    assert len(best["history"]) == 3


def test_respects_max_rounds():
    best = optimize("t", PAD3, rounds=1, swarm_fn=fake_swarm({PAD3: PAD1, PAD1: GOAL}))
    assert best["qasm"] == PAD1 and best["rounds_run"] == 1


def test_never_accepts_non_equivalent():
    # Round 1 is fine; round 2 offers a smaller but wrong circuit. Must be rejected.
    best = optimize("t", PAD3, rounds=5, swarm_fn=fake_swarm({PAD3: PAD1, PAD1: BAD}))
    assert best["qasm"] == PAD1
    assert tools.check_equiv(PAD3, best["qasm"])["ok"]


def test_equivalence_checked_against_original():
    seen = []
    from quantum_optimizer.reviewer import review
    def spy(before, cands, **kw):
        seen.append(kw.get("reference"))
        return review(before, cands, **kw)
    optimize("t", PAD3, rounds=5, swarm_fn=fake_swarm({PAD3: PAD1, PAD1: GOAL}), review_fn=spy)
    assert seen and all(r == PAD3 for r in seen)


def test_picker_gets_round_and_chain():
    calls = []
    def picker(name, qasm, r, chain):
        calls.append((r, tuple(chain)))
        return ["peel"]
    optimize("t", PAD3, picker=picker, rounds=5, swarm_fn=fake_swarm({PAD3: PAD1, PAD1: GOAL}))
    assert calls[:2] == [(1, ()), (2, ("peel",))]


def test_pick_keeps_planner_order(monkeypatch):
    monkeypatch.setattr(orchestrator, "plan", lambda *a, **k: "try level2 first, then level1")
    assert orchestrator.pick("x", GOAL, "meta") == ["level2", "level1"]


def test_grover_chain_reaches_81_offline():
    q = all_benchmarks()["grover3"]
    best = optimize("grover3", q, rounds=4)
    assert tools.check_equiv(q, best["qasm"])["ok"]
    assert best["total"] <= 81 and len(best["chain"]) >= 2
