"""GL-free tests: camera controller, culling, LOD, batching, instance-cache rebuilds."""
import math

import numpy as np

from quantum_optimizer import scene3d as S, viewer
from tests.test_scene3d import _state


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def _ctl():
    clk = Clock(); rig = S.CameraRig(); rig.dist = 20.0
    return S.CameraController(rig, clock=clk), clk, rig


def _run(c, clk, secs, dt=1 / 60):
    for _ in range(int(secs / dt)):
        clk.t += dt; c.rig.t += dt; c.update(dt)


def test_drag_changes_yaw_pitch_and_pitch_clamps():
    c, clk, rig = _ctl()
    _, y0, p0, _ = c.pose()
    c.drag(100, 40); _run(c, clk, 1.0)
    _, y1, p1, _ = c.pose()
    assert y1 < y0 - 0.3 and p1 > p0 + 0.1
    c.drag(0, 10_000); _run(c, clk, 1.0)
    assert math.isclose(c.pose()[2], S.PITCH_MAX, abs_tol=1e-3)
    c.drag(0, -100_000); _run(c, clk, 1.0)
    assert math.isclose(c.pose()[2], S.PITCH_MIN, abs_tol=1e-3)


def test_zoom_is_exponential_and_clamped():
    c, clk, _ = _ctl()
    c.zoom(1); _run(c, clk, 1.0)
    assert math.isclose(c.pose()[3], 20 * math.exp(-0.12), rel_tol=1e-3)
    c.zoom(500); _run(c, clk, 1.0)
    assert math.isclose(c.pose()[3], S.DIST_MIN, rel_tol=1e-2)
    c.zoom(-5000); _run(c, clk, 1.0)
    assert math.isclose(c.pose()[3], S.DIST_MAX, rel_tol=1e-2)


def test_pan_moves_target():
    c, clk, _ = _ctl()
    t0 = c.pose()[0].copy()
    for _ in range(30):
        c.pan(1, 0, 1 / 60); clk.t += 1 / 60; c.update(1 / 60)
    _run(c, clk, 0.5)
    assert np.linalg.norm(c.pose()[0] - t0) > 1.0


def test_glide_paused_during_input_then_resumes_smoothly():
    c, clk, rig = _ctl()
    c.drag(200, 0); _run(c, clk, 1.0)
    assert c.w > 0.99 and c.user_active
    user_yaw = c.pose()[1]
    _run(c, clk, 1.5)                       # rig keeps gliding but the user pose holds
    assert math.isclose(c.pose()[1], user_yaw, abs_tol=5e-3)  # glide yaw moves ~0.03 here
    _run(c, clk, 1.0)                       # > 3 s idle: blend back starts
    assert not c.user_active
    prev, jumps = c.pose()[1], []
    for _ in range(600):
        clk.t += 1 / 60; rig.t += 1 / 60; c.update(1 / 60)
        y = c.pose()[1]; jumps.append(abs(y - prev)); prev = y
    assert max(jumps) < 0.02                # no snap
    assert c.w < 0.01 and math.isclose(c.pose()[1], rig.pose()[1], abs_tol=0.02)


def test_reset_returns_to_glide():
    c, clk, rig = _ctl()
    c.zoom(5); c.drag(300, 50); _run(c, clk, 0.5)
    c.reset()
    assert not c.user_active
    w_after_reset = c.w; _run(c, clk, 0.05)
    assert c.w < w_after_reset              # blending, not snapping
    _run(c, clk, 6.0)
    assert c.w < 0.01 and math.isclose(c.pose()[3], rig.pose()[3], rel_tol=0.02)


def _vp(eye=(0, 3, 15), at=(0, 0, 0)):
    return S.perspective(48, 16 / 9, 0.1, 600) @ S.look_at(eye, at)


def test_frustum_culling():
    planes = S.frustum_planes(_vp())
    pts = np.array([[0, 0, 0], [0, 0, 40], [200, 0, -10], [0, 0, -500], [0, 0, -700.0]])
    m = S.visible_mask(pts, np.full(len(pts), 1.0), planes)
    assert m.tolist() == [True, False, False, True, False]   # in, behind, side, far-in, beyond far


def test_cull_lod_and_batching_counts():
    st = _state(); ic = S.InstanceCache()
    ic.refresh(st, 10.0, viewer.Look())
    nc, nf, nr = ic.cull(_vp(eye=(0, 4, 14), at=(0, 0, -5)), (0, 4, 14))
    assert (nc, nf, nr) == (8, 2, 3)       # all cubes, flames only for the 2 working cubes, 3 ribbons
    far = (0, 4, S.LOD_FAR + 30)
    nc2, nf2, _ = ic.cull(_vp(eye=far, at=(0, 0, -5)), far)
    assert nc2 == 8 and nf2 == 0            # LOD: far working cubes lose their flames
    nc3, nf3, nr3 = ic.cull(_vp(eye=(0, 4, 14), at=(0, 4, 60)), (0, 4, 14))  # looking away
    assert (nc3, nf3, nr3) == (0, 0, 0)
    assert ic.vis_cubes.shape[0] >= 8      # preallocated buffer reused


def test_instance_buffers_rebuild_only_on_change():
    st = _state(); L = viewer.Look(); ic = S.InstanceCache()
    st.gates["work"].work_until = -1        # nothing animating
    for g in st.gates.values():
        g.spin = 0.0
    st.version = 1
    assert ic.refresh(st, 10.0, L) is True
    assert ic.refresh(st, 10.1, L) is False and ic.refresh(st, 10.2, L) is False
    assert ic.builds == 1
    st.version += 1                          # an event arrived
    assert ic.refresh(st, 10.3, L) is True and ic.builds == 2
    st.gates["h"].alpha = 0.5                # fading in: keep rebuilding while animating
    st.version += 1
    ic.refresh(st, 10.4, L)
    assert ic.refresh(st, 10.5, L) is True


def test_internal_scale_retina_aware():
    from quantum_optimizer.gl3d import auto_scale
    assert auto_scale((1440, 900), (2880, 1800)) == 0.5
    assert auto_scale((1600, 900), (1600, 900)) == 0.75
