"""Post-run wind-down: removed/failed frames hold, fade, then drop; live gates untouched. No GPU / network."""
from quantum_optimizer import scene3d, viewer
from quantum_optimizer.viewer import FAILED, IDLE, REMOVED, State


def g(name, q, layer):
    return {"name": name, "qubits": q, "layer": layer}


START = [g("h", [0], 0), g("x", [1], 0), g("cx", [0, 1], 1), g("t", [0], 2)]
AFTER = [g("h", [0], 0), g("cx", [0, 1], 1)]  # x and t removed


def run(st, hold=12.0, fade=3.0):
    st.configure_winddown(hold, fade)
    st.apply({"type": "run_start", "circuit": "t", "gate_list": START, "n_qubits": 2, "gates": 4}, 0.0)
    st.apply({"type": "circuit_update", "circuit_id": "c1", "gate_list": AFTER, "n_qubits": 2, "gates": 2}, 1.0)
    st.apply({"type": "result", "source": "chain", "equiv": True}, 5.0)
    return st


def states(st, now):
    return sorted(x.state(now) for x in st.gates.values())


def tick_to(st, t):
    st.tick(1 / 60, t)


def test_hold_keeps_frames_fully_visible_with_countdown():
    st = run(State())
    for t in (5.0, 10.0, 16.9):
        tick_to(st, t)
        assert st.wd_k == 1.0
        assert states(st, t).count(REMOVED) == 2
    tick_to(st, 8.2)
    assert "clear in 9s" in st.wd_text


def test_fade_is_monotonic_ease_out_and_reaches_scene():
    st = run(State())
    ks = []
    for t in (17.5, 18.5, 19.5):
        tick_to(st, t)
        ks.append(st.wd_k)
    assert 1.0 > ks[0] > ks[1] > ks[2] > 0.0
    assert ks[0] < 1 - 0.5 / 3  # ease-out: drops faster than linear at the start
    assert states(st, 19.5).count(REMOVED) == 2  # still there while fading
    L = viewer.look()
    cubes, _ = scene3d.build(st, 19.5, L)
    removed = cubes[cubes[:, 4] == scene3d.STATE_CODE[REMOVED]]
    assert len(removed) and (removed[:, 6] < L.still[1] * 0.2).all()


def test_gone_after_fade_live_gates_unaffected():
    st = run(State())
    live_before = {k for k, x in st.gates.items() if x.state(5.0) not in (REMOVED, FAILED)}
    v = st.version
    tick_to(st, 20.5)
    assert REMOVED not in states(st, 20.5)
    assert set(st.gates) == live_before
    assert st.version > v
    assert all(x.alpha > 0 for x in st.gates.values())
    cubes, _ = scene3d.build(st, 20.5, viewer.look())
    assert (cubes[:, 6] > 0).all() and len(cubes) == 3  # h + cx(2 qubits)


def test_failed_frames_also_clear():
    st = State(); st.configure_winddown(2.0, 1.0)
    st.apply({"type": "run_start", "gate_list": START, "n_qubits": 2}, 0.0)
    st.apply({"type": "circuit_update", "circuit_id": "c1", "gate_list": AFTER, "n_qubits": 2}, 1.0)
    st.apply({"type": "equivalence", "ok": False, "circuit_id": "c1"}, 1.1)
    st.restore_at = None  # keep them failed
    st.apply({"type": "run_end"}, 2.0)
    tick_to(st, 3.0)
    assert FAILED in states(st, 3.0)
    tick_to(st, 5.5)
    assert st.gates == {}


def test_new_run_start_cancels_and_resets():
    st = run(State())
    tick_to(st, 18.5)
    assert st.wd_k < 1.0
    st.apply({"type": "run_start", "gate_list": START, "n_qubits": 2}, 18.6)
    assert st.wd_start is None and st.wd_k == 1.0 and st.wd_text == ""
    st.apply({"type": "circuit_update", "circuit_id": "c1", "gate_list": AFTER, "n_qubits": 2}, 19.0)
    tick_to(st, 40.0)  # no result yet: nothing clears
    assert states(st, 40.0).count(REMOVED) == 2


def test_disabled_and_no_end_signal_keep_frames():
    st = State(); st.configure_winddown(enabled=False)
    run(st); st.configure_winddown(enabled=False)
    st.wd_start = None
    st.apply({"type": "run_end"}, 6.0)
    tick_to(st, 100.0)
    assert states(st, 100.0).count(REMOVED) == 2


def test_playback_finished_starts_winddown():
    st = State(); st.configure_winddown(1.0, 1.0)
    st.apply({"type": "run_start", "gate_list": START, "n_qubits": 2}, 0.0)
    st.apply({"type": "circuit_update", "circuit_id": "c1", "gate_list": AFTER, "n_qubits": 2}, 1.0)
    st.apply({"type": "_status", "text": "playback finished"}, 2.0)
    tick_to(st, 4.1)
    assert IDLE in states(st, 4.1) and REMOVED not in states(st, 4.1)


def test_highlight_ramps_up_in_hold_and_brightens_frames():
    st = run(State())
    L = viewer.look()
    cubes0, _ = scene3d.build(st, 4.9, L)
    rem = scene3d.STATE_CODE[REMOVED]
    assert st.wd_hl == 0.0 and (cubes0[cubes0[:, 4] == rem][:, 11] == 0).all()
    tick_to(st, 5.25)
    assert 0.0 < st.wd_hl < 1.0
    tick_to(st, 6.0)
    assert st.wd_hl == 1.0
    cubes, _ = scene3d.build(st, 6.0, L)
    r = cubes[cubes[:, 4] == rem]
    assert (r[:, 6] > 0.99).all() and (r[:, 11] == 1.0).all()          # full alpha, highlight flag
    assert r[:, 7:10].mean() > cubes0[cubes0[:, 4] == rem][:, 7:10].mean()  # brighter colour
    assert (r[:, 10] == scene3d.STYLE["dashed"]).all()                   # keeps its dash pattern
    assert (r[:, 8] < 0.95).all() or (r[:, 7] > 0.8).all()               # cool grey, not turquoise
    live = cubes[cubes[:, 4] != rem]
    live0 = cubes0[cubes0[:, 4] != rem]
    assert (live[:, 6:11] == live0[:, 6:11]).all()                       # live gates unaffected


def test_highlight_resets_on_new_run_and_frames_gone_after_fade():
    st = run(State())
    tick_to(st, 6.0)
    assert st.wd_hl == 1.0
    tick_to(st, 20.5)
    assert REMOVED not in states(st, 20.5)
    st.apply({"type": "run_start", "gate_list": START, "n_qubits": 2}, 21.0)
    assert st.wd_hl == 0.0
