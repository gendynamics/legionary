"""GL-free tests for the cinematic 3D scene logic (scene3d)."""
import numpy as np

from quantum_optimizer import scene3d as S, viewer, visual_director as vd


def _state(now=10.0):
    st = viewer.State()
    st.nq = 3
    specs = {"h": ("h", [0], 0), "cx": ("cx", [0, 2], 1), "gone": ("x", [1], 2),
             "bad": ("cx", [1, 2], 3), "work": ("cx", [0, 1], 4)}
    for k, (n, q, l) in specs.items():
        st.gates[k] = viewer.Gate(k, {"name": n, "qubits": q, "layer": l}, now, fresh=False)
    st.gates["gone"].fail(now - 1)
    st.gates["bad"].eq_failed = True
    st.gates["work"].work_until = now + 1.0
    return st


def _build(now=10.0):
    return S.build(_state(now), now, viewer.Look())


def test_instance_shapes_and_positions():
    c, r = _build()
    assert c.shape == (8, S.CUBE_FIELDS) and r.shape == (3, S.RIBBON_FIELDS)
    h = c[0]
    assert np.allclose(h[:3], [-S.GAP_Q, 0, 0])
    gone = c[c[:, 4] == S.STATE_CODE[viewer.REMOVED]][0]
    assert gone[1] == S.REMOVED_DROP and np.isclose(gone[2], -2 * S.GAP_L) and gone[0] == 0
    assert np.isclose(c[1, 3], S.CUBE * 1.12) and np.isclose(c[0, 3], S.CUBE)  # cx cubes larger


def test_state_flags_and_styles():
    c, _ = _build()
    codes = sorted(set(c[:, 4].astype(int)))
    assert codes == [0, 1, 2, 3]  # idle, working, failed, removed all still on screen
    removed = c[c[:, 4] == 3]; failed = c[c[:, 4] == 2]; working = c[c[:, 4] == 1]
    assert (removed[:, 10] == S.STYLE["dashed"]).all()
    assert (failed[:, 10] == S.STYLE["dotted"]).all()
    assert (working[:, 11] > 0).all() and (c[c[:, 4] == 0][:, 11] == 0).all()


def test_ribbon_widths():
    L = viewer.Look()
    idle = S.ribbon_width(L, viewer.IDLE, 0.0)
    work = S.ribbon_width(L, viewer.WORKING, 1.0)
    assert work > idle > 0
    assert S.ribbon_width(L, viewer.REMOVED, 1.0) == 0.04 == S.ribbon_width(L, viewer.FAILED, 0.0)
    _, r = _build()
    assert r[r[:, 7] == 1][0, 6] == max(r[:, 6])  # the working CX ribbon is the widest


def test_camera_pushes_toward_working():
    st = _state(); c, _ = S.build(st, 10.0, viewer.Look())
    tgt, dist = S.CameraRig().goal(c, st)
    assert np.isclose(tgt[2], -4 * S.GAP_L)  # frames the working gate's layer
    st2 = _state(); st2.gates["work"].work_until = -1
    c2, _ = S.build(st2, 10.0, viewer.Look())
    _, dist_idle = S.CameraRig().goal(c2, st2)
    assert dist < dist_idle
    rig = S.CameraRig()
    for _ in range(200):
        rig.update(1 / 30, c, st)
    assert np.linalg.norm(rig.target - tgt) < 0.5


def test_projection_shift_and_look_at():
    p0 = S.perspective(40, 16 / 9, 0.1, 100)
    p1 = S.perspective(40, 16 / 9, 0.1, 100, shift_x=-0.2)
    v = np.array([0, 0, -10, 1.0])
    a, b = p0 @ v, p1 @ v
    assert np.isclose(a[0] / a[3], 0) and np.isclose(b[0] / b[3], -0.2)
    m = S.look_at((0, 0, 5), (0, 0, 0))
    assert np.allclose(m @ np.array([0, 0, 0, 1.0]), [0, 0, -5, 1])


def test_director_defaults_turquoise_and_scene_clamp():
    assert vd.DEFAULTS["flame"]["mid"].lower() == "#40e0d0"
    assert vd.TIMEOUT >= 45
    th, _ = vd.validate({"scene": {"bloom": 9, "fog": -1, "camera_speed": 0.01, "exposure": 1.3}})
    assert th["scene"] == {"bloom": 2.5, "fog": 0.0, "camera_speed": 0.2, "exposure": 1.3}
