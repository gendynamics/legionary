import json, socket, time
from quantum_optimizer import events, orchestrator, viewer
from quantum_optimizer.benchmarks import all_benchmarks
from quantum_optimizer.agent_planner import Session


def test_disabled_by_default_is_noop(tmp_path):
    events.close()
    assert not events.enabled()
    events.emit("tool_call", call=1)  # must not raise


def test_snapshot_layers():
    q = all_benchmarks()["grover3"]
    s = events.snapshot(q)
    assert s["gates"] == len(s["gate_list"]) and s["n_qubits"] == 3
    assert max(g["layer"] for g in s["gate_list"]) + 1 == s["depth"]


def test_chain_run_writes_jsonl(tmp_path):
    p = tmp_path / "ev.jsonl"
    orchestrator.main(["--planner", "chain", "--only", "bv4", "--no-nemotron-review", "--events", str(p)])
    evs = [json.loads(l) for l in p.read_text().splitlines()]
    types = [e["type"] for e in evs]
    assert types[0] == "run_start" and "result" in types and types[-1] == "run_end"
    assert not events.enabled()


def test_session_dispatch_emits(tmp_path):
    p = tmp_path / "s.jsonl"
    events.configure(path=str(p))
    try:
        s = Session(all_benchmarks()["grover3"], max_calls=5)
        r = s.dispatch("run_pass", {"circuit_id": "c0", "strategy": "level1"})
        s.dispatch("check_equivalence", {"circuit_id": r["circuit_id"]})
    finally:
        events.close()
    types = [json.loads(l)["type"] for l in p.read_text().splitlines()]
    assert types == ["tool_call", "circuit_update", "tool_call", "equivalence"]


def test_tcp_backlog_replay():
    em = events.configure(port=0)
    try:
        events.emit("run_start", circuit="x")
        c = socket.create_connection(("127.0.0.1", em.port), timeout=3)
        time.sleep(0.2)
        events.emit("tool_call", call=1)
        data = b""
        while data.count(b"\n") < 2:
            data += c.recv(4096)
        c.close()
    finally:
        events.close()
    assert [json.loads(l)["type"] for l in data.splitlines()] == ["run_start", "tool_call"]


def test_viewer_state_from_demo():
    st = viewer.State()
    for i, ev in enumerate(viewer.demo_events()):
        st.apply(ev, float(i))
        st.tick(0.1, float(i))
    assert st.demo and st.done and st.equiv is True and st.now_gates == 81 and st.before == 157


def _ev(gl, t="circuit_update", **kw):
    return {"type": t, "gates": len(gl), "depth": 1 + max(g["layer"] for g in gl), "cx": 0,
            "n_qubits": 2, "gate_list": gl, **kw}


A = {"name": "rz", "qubits": [0], "layer": 0}
B = {"name": "sx", "qubits": [1], "layer": 0}
C = {"name": "x", "qubits": [0], "layer": 1}


def test_new_gates_are_working_then_settle():
    st = viewer.State()
    st.apply(_ev([A, B], "run_start"), 0.0)
    assert not st.working(0.0)  # initial circuit is idle
    st.apply(_ev([A, B, C], circuit_id="c1"), 1.0)
    w = st.working(1.1)
    assert [g.name for g in w] == ["x"]
    st.tick(0.1, 1.1)
    assert w[0].spin > 0  # inner stratosphere spins
    later = 1.0 + viewer.WORK_SECS + 0.1
    st.tick(0.1, later)
    assert not st.working(later) and w[0].state(later) == viewer.IDLE


def test_removed_gate_stays_on_map_static():
    st = viewer.State()
    st.apply(_ev([A, B], "run_start"), 0.0)
    st.apply(_ev([A], circuit_id="c1"), 1.0)
    gb = [g for g in st.gates.values() if g.name == "sx"][0]
    for t in (1.1, 3.0, 30.0):
        st.tick(0.1, t)
    assert gb.key in st.gates and gb.state(30.0) == viewer.REMOVED  # never dropped
    assert gb.dissolve(30.0) == 0 and gb.spin == 0 and gb.alpha == 1.0
    assert gb not in st.working(30.0) and len(st.visible(30.0)) == 2 and len(st.current(30.0)) == 1
    assert st.now_gates == 1  # ticker = current circuit only
    st.apply(_ev([A, B], circuit_id="c2"), 31.0)  # comes back -> lit and working again
    assert gb.state(31.1) == viewer.WORKING


def test_window_frame_is_turquoise():
    assert viewer.TURQUOISE == (0x40, 0xE0, 0xD0)


def test_equivalence_fail_keeps_cubes_static_then_restores():
    st = viewer.State()
    st.apply(_ev([A, B], "run_start"), 0.0)
    st.apply(_ev([A, C], circuit_id="c1"), 1.0)
    st.apply({"type": "equivalence", "circuit_id": "c1", "ok": False}, 1.2)
    states = {g.name: g.state(1.3) for g in st.gates.values()}
    assert states["x"] == viewer.FAILED and states["rz"] == viewer.FAILED
    assert not st.working(1.3)  # failed cubes: no spin, no flames
    spin = [g for g in st.gates.values() if g.name == "x"][0].spin
    st.tick(0.1, 1.4)
    assert [g for g in st.gates.values() if g.name == "x"][0].spin == spin and st.equiv is False
    t = 1.2 + viewer.HOLD_FAIL_SECS + 0.01
    st.tick(0.1, t)  # last passing circuit (A, B) comes back
    alive = {g.name for g in st.gates.values() if g.state(t) != viewer.REMOVED}
    assert alive == {"rz", "sx"}
    assert "x" in {g.name for g in st.visible(t)}  # the failed candidate's extra gate stays on the map
