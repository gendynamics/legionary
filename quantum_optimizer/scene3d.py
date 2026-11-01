"""3D scene logic for the cinematic GPU viewer (no GL here, so it is unit-testable).

World layout (grand scale): qubits are rows side by side along x, circuit layers recede into the
scene along -z, cubes float at y=0. Removed cubes stay as still frames, sunk below the live plane;
failed cubes stay in place. build() turns viewer.State into flat instance arrays for instancing."""
import math

import numpy as np

from .viewer import IDLE, WORKING, FAILED, REMOVED, HL_FAILED, HL_REMOVED, highlight

GAP_Q, GAP_L = 3.4, 2.7         # world units between qubit rows / circuit layers
CUBE = 0.85                      # cube half-size (CX cubes x1.12)
REMOVED_DROP = -2.4              # removed cubes sink below the live plane
STATE_CODE = {IDLE: 0, WORKING: 1, FAILED: 2, REMOVED: 3}
STYLE = {"solid": 0, "dashed": 1, "dotted": 2}
CUBE_FIELDS = 12   # x y z | half | state | spin | alpha | r g b | style | work
RIBBON_FIELDS = 12  # x0 y0 z0 | x1 y1 z1 | width | activity | alpha | state | arc | phase


def gate_pos(st, g, state):
    x = (g.qubits[0] - (st.nq - 1) / 2) * GAP_Q
    z = -g.layer * GAP_L
    return x, (REMOVED_DROP if state == REMOVED else 0.0), z


def build(st, now, look):
    """-> (cubes [N,12] float32, ribbons [M,12] float32). One cube per (gate, qubit)."""
    cubes, ribbons = [], []
    anim = False
    still_c, still_a, still_s = look.still
    fail_c, fail_a, fail_s = look.failed
    wk, hl = getattr(st, "wd_k", 1.0), getattr(st, "wd_hl", 0.0)  # post-run wind-down highlight + fade
    still_c, still_a = highlight(still_c, still_a, hl, HL_REMOVED)
    fail_c, fail_a = highlight(fail_c, fail_a, hl, HL_FAILED)
    still_a, fail_a = still_a * wk, fail_a * wk
    for g in st.gates.values():
        state = g.state(now)
        if state != REMOVED and (g.alpha < 0.999 or abs(g.target - g.layer) > 1e-3 or state == WORKING
                                 or abs(g.spin - round(g.spin / (math.pi / 2)) * (math.pi / 2)) > 1e-3):
            anim = True
        work = max(0.0, min(1.0, (g.work_until - now) / 0.6)) if state == WORKING else 0.0
        if state in (REMOVED, FAILED):
            work = hl  # the 'work' slot carries the wind-down highlight for empty frames (no new attribute)
        if state == REMOVED:
            col, alpha, style = still_c, still_a, still_s
        elif state == FAILED:
            col, alpha, style = fail_c, fail_a, fail_s
        else:
            col, alpha, style = look.gate_edge(g.name), g.alpha, "solid"
        half = CUBE * (1.12 if g.name == "cx" else 1.0)
        _, y, z = gate_pos(st, g, state)
        for q in g.qubits:
            x = (q - (st.nq - 1) / 2) * GAP_Q
            cubes.append([x, y, z, half, STATE_CODE[state], g.spin, alpha,
                          *(v / 255 for v in col), STYLE.get(style, 0), work])
        if len(g.qubits) == 2:
            a, b = g.qubits
            xa, xb = ((q - (st.nq - 1) / 2) * GAP_Q for q in (a, b))
            act = 1.0 if state == WORKING else 0.0
            ribbons.append([xa, y, z, xb, y, z, ribbon_width(look, state, act), act,
                            alpha if state in (REMOVED, FAILED) else g.alpha, STATE_CODE[state],
                            0.35 * abs(b - a), (hash(g.key) % 997) / 997])
    c = np.array(cubes, np.float32).reshape(-1, CUBE_FIELDS)
    r = np.array(ribbons, np.float32).reshape(-1, RIBBON_FIELDS)
    build.animating = anim
    return c, r


build.animating = True


def ribbon_width(look, state, activity):
    """World-space ribbon width: thin for idle, base->max while the gate is worked on, hairline
    when the CX gate was removed/failed."""
    if state in (REMOVED, FAILED):
        return 0.04
    k = 0.035  # theme widths are in 'pixels' (1-16); scale to world units
    return k * (look.ribbon_w + (look.ribbon_max - look.ribbon_w) * activity)


class CameraRig:
    """Low cinematic camera: slow orbit + dolly; frames the active region and pushes in toward
    working cubes. Everything is smoothed so it never jumps."""

    def __init__(self, angle_deg=0.0):
        self.target = np.zeros(3)
        self.dist, self.yaw0 = 22.0, math.radians(angle_deg)
        self.t = 0.0

    def goal(self, cubes, st):
        live = cubes[cubes[:, 4] != STATE_CODE[REMOVED]] if len(cubes) else cubes
        working = cubes[cubes[:, 4] == STATE_CODE[WORKING]] if len(cubes) else cubes
        span_x = max(st.nq - 1, 1) * GAP_Q
        if len(working):
            tgt = working[:, :3].mean(0)
            tgt[2] = np.clip(tgt[2], live[:, 2].min() if len(live) else tgt[2], 0)
            dist = max(span_x * 3.0, 18.0)
        elif len(live):
            tgt = np.array([0.0, 0.0, live[:, 2].min() * 0.3])
            dist = max(span_x * 3.8, 22.0)
        else:
            tgt, dist = np.zeros(3), 18.0
        tgt[1] = 0.0
        return tgt, dist

    def update(self, dt, cubes, st, speed=1.0):
        self.t += dt * speed
        tgt, dist = self.goal(cubes, st)
        k = min(1.0, dt * 0.8 * speed)
        self.target += (tgt - self.target) * k
        self.dist += (dist - self.dist) * k * 0.6

    def pose(self):
        """Glide pose as (target xyz, yaw, pitch, dist): the CameraController blends these params."""
        yaw = self.yaw0 + 0.42 * math.sin(self.t * 0.045) + 0.18
        el = math.radians(17 + 4 * math.sin(self.t * 0.07))
        return np.array(self.target, float), yaw, el, float(self.dist)

    def eye(self):
        return eye_from_pose(*self.pose())


def eye_from_pose(target, yaw, el, d):
    off = np.array([math.sin(yaw) * math.cos(el), math.sin(el), math.cos(yaw) * math.cos(el)]) * d
    look_at = target + np.array([0.0, 0.0, -d * 0.35])  # look slightly down-field
    return target + off, look_at


def perspective(fovy_deg, aspect, near, far, shift_x=0.0):
    f = 1 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    m[0, 2] = -shift_x  # off-axis: x_ndc += shift_x (keeps the scene centred left of the panel)
    return m


def look_at(eye, center, up=(0, 1, 0)):
    eye, center, up = (np.asarray(v, np.float64) for v in (eye, center, up))
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


# ---------------------------------------------------------------- interactive camera
PITCH_MIN, PITCH_MAX = math.radians(-8), math.radians(80)
DIST_MIN, DIST_MAX = 4.0, 160.0
IDLE_RESUME = 3.0       # seconds without input before the auto-glide takes over again


def _damp(cur, goal, rate, dt):
    """Exponential (critically damped first-order) approach, frame-rate independent."""
    return goal + (cur - goal) * math.exp(-rate * dt)


class CameraController:
    """User camera on top of the CameraRig glide. Inputs set *goals* (yaw, pitch, dist, pan); the
    shown pose chases them exponentially. `w` blends glide (0) -> user (1): any input sets w's goal
    to 1 (the user pose is seeded from the current shown pose, so there is no snap); after
    IDLE_RESUME s idle w's goal is 0 and the view eases back onto the moving glide path. R resets:
    clears the pan/zoom offsets and hands back to the glide at once (still blended)."""

    def __init__(self, rig, clock=None):
        import time as _t
        self.rig, self.clock = rig, clock or _t.monotonic
        self.w = self.w_goal = 0.0
        self.last_input = -1e9
        self.yaw = self.pitch = 0.0; self.dist = 20.0; self.target = np.zeros(3)
        self.g_yaw = self.g_pitch = 0.0; self.g_dist = 20.0; self.g_target = np.zeros(3)
        self._shown = None

    # -- inputs
    def _touch(self):
        if self.w < 0.02 or self._shown is None:  # seed from what is on screen now
            tg, yaw, pitch, dist = self._shown or self.rig.pose()
            self.yaw = self.g_yaw = yaw; self.pitch = self.g_pitch = pitch
            self.dist = self.g_dist = dist; self.target = np.array(tg, float); self.g_target = self.target.copy()
        elif self.w_goal == 0.0:  # mid-blend back to glide: continue from the shown pose
            tg, yaw, pitch, dist = self._shown
            self.yaw, self.pitch, self.dist, self.target = yaw, pitch, dist, np.array(tg, float)
        self.w_goal = 1.0
        self.last_input = self.clock()

    def drag(self, dx, dy, sens=0.005):
        self._touch()
        self.g_yaw -= dx * sens
        self.g_pitch = min(PITCH_MAX, max(PITCH_MIN, self.g_pitch + dy * sens))

    def zoom(self, steps):
        self._touch()
        self.g_dist = min(DIST_MAX, max(DIST_MIN, self.g_dist * math.exp(-0.12 * steps)))

    def pan(self, right, fwd, dt):
        self._touch()
        k = self.g_dist * 0.8 * dt
        r = np.array([math.cos(self.g_yaw), 0.0, -math.sin(self.g_yaw)])
        f = np.array([-math.sin(self.g_yaw), 0.0, -math.cos(self.g_yaw)])
        self.g_target = self.g_target + (r * right + f * fwd) * k

    def reset(self):
        self.w_goal = 0.0
        self.last_input = -1e9

    @property
    def user_active(self):
        return self.w_goal > 0.5

    # -- per frame
    def update(self, dt):
        if self.w_goal > 0 and self.clock() - self.last_input > IDLE_RESUME:
            self.w_goal = 0.0
        self.yaw = _damp(self.yaw, self.g_yaw, 10, dt); self.pitch = _damp(self.pitch, self.g_pitch, 10, dt)
        self.dist = _damp(self.dist, self.g_dist, 8, dt)
        self.target = self.g_target + (self.target - self.g_target) * math.exp(-8 * dt)
        self.w = _damp(self.w, self.w_goal, 6 if self.w_goal > self.w else 1.2, dt)
        if abs(self.w - self.w_goal) < 3e-3:
            self.w = self.w_goal
        return self.pose()

    def pose(self):
        gt, gy, gp, gd = self.rig.pose()
        w = self.w
        dy = math.atan2(math.sin(self.yaw - gy), math.cos(self.yaw - gy))  # shortest way round
        p = (gt + (self.target - gt) * w, gy + dy * w, gp + (self.pitch - gp) * w,
             math.exp(math.log(gd) + (math.log(self.dist) - math.log(gd)) * w))
        self._shown = p
        return p

    def eye(self):
        return eye_from_pose(*self.pose())


# ---------------------------------------------------------------- culling, LOD, instance cache
LOD_FAR = 55.0    # beyond this distance: no flame ring, edges drawn without dash/dot patterns


def frustum_planes(vp):
    """6 normalized planes (a,b,c,d) of clip space from a view-projection matrix (Gribb-Hartmann)."""
    m = np.asarray(vp, np.float64)
    pl = np.array([m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1], m[3] + m[2], m[3] - m[2]])
    return pl / np.linalg.norm(pl[:, :3], axis=1, keepdims=True)


def visible_mask(centers, radii, planes):
    """Sphere-vs-frustum: kept if signed distance >= -r for all six planes (vectorized)."""
    if len(centers) == 0:
        return np.zeros(0, bool)
    d = centers @ planes[:, :3].T + planes[:, 3]
    return (d >= -radii[:, None]).all(1)


class InstanceCache:
    """Builds instance arrays only when the circuit changed (State.version) or something is still
    animating; then culls into preallocated buffers each frame. Exposes counts for the draw calls:
    one instanced call each for spheres, edges (dashed/dotted via the style attribute), ribbons and
    flames (near idle + working; idle at pilot-light strength)."""

    def __init__(self, cap=256):
        self.version = None
        self.builds = 0
        self.cubes = np.zeros((0, CUBE_FIELDS), np.float32)
        self.ribbons = np.zeros((0, RIBBON_FIELDS), np.float32)
        self._alloc(cap)

    def _alloc(self, cap):
        self.cap = cap
        self.vis_cubes = np.zeros((cap, CUBE_FIELDS), np.float32)
        self.vis_flames = np.zeros((cap, CUBE_FIELDS), np.float32)
        self.vis_ribs = np.zeros((cap, RIBBON_FIELDS), np.float32)

    def refresh(self, st, now, look):
        v = getattr(st, "version", 0)
        if v == self.version and not self._anim:
            return False
        self.cubes, self.ribbons = build(st, now, look)
        self._anim = build.animating
        self.version = v
        self.builds += 1
        need = max(len(self.cubes), len(self.ribbons))
        if need > self.cap:
            self._alloc(1 << int(math.ceil(math.log2(need))))
        return True

    _anim = True

    def cull(self, vp, eye):
        """-> (n_cubes, n_flames, n_ribbons) written to the front of the preallocated buffers."""
        planes, eye = frustum_planes(vp), np.asarray(eye, np.float64)
        c, r = self.cubes, self.ribbons
        m = visible_mask(c[:, :3].astype(np.float64), c[:, 3] * 1.9, planes) if len(c) else np.zeros(0, bool)
        nc = int(m.sum()); self.vis_cubes[:nc] = c[m]
        if len(c):
            dist = np.linalg.norm(c[:, :3] - eye, axis=1)
            fm = m & (c[:, 4] <= STATE_CODE[WORKING]) & (dist < LOD_FAR)
        else:
            fm = m
        nf = int(fm.sum()); self.vis_flames[:nf] = c[fm]
        if len(r):
            mid = (r[:, 0:3] + r[:, 3:6]) * 0.5
            rad = np.linalg.norm(r[:, 3:6] - r[:, 0:3], axis=1) * 0.5 + r[:, 10] + 0.5
            rm = visible_mask(mid.astype(np.float64), rad, planes)
        else:
            rm = np.zeros(0, bool)
        nr = int(rm.sum()); self.vis_ribs[:nr] = r[rm]
        self.counts = (nc, nf, nr)
        return self.counts
