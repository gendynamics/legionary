"""Legionary live viewer: a native pygame window (no HTML). Near-black void; each gate is a wireframe
cube with a stratosphere sky inside it. Working gates spin inside a ring of flames; removed/failed gates
stay as still frames. Colors and styles come from a theme chosen once at startup by Muse Spark (the
'visual director', see visual_director.py); built-in defaults are used until/unless it arrives.

  python -m quantum_optimizer.viewer --connect 127.0.0.1:8765   # live, from orchestrator --events tcp:8765
  python -m quantum_optimizer.viewer --replay run.jsonl          # play back a saved stream
  python -m quantum_optimizer.viewer --demo                      # synthetic Grover-like run (labeled DEMO)
  --director-backend {auto,meta,fireworks,nebius,off}  --director-refresh (ignore cached theme)
Keys: Esc/q quit, space pause (replay/demo).
"""
import argparse, json, math, os, queue, random, socket, sys, threading, time

import numpy as np

from . import visual_director as vd

W, H = 1280, 720
WHITE = (255, 255, 255)


class Look:
    """Theme (visual_director JSON) resolved to RGB tuples / floats for drawing. The viewer has no
    palette of its own: everything below comes from the theme (vd.DEFAULTS when no model theme)."""

    def __init__(self, theme=None, label="theme: default"):
        t = theme or vd.DEFAULTS
        c = vd.hex_rgb
        self.theme, self.label = t, label
        for k, v in t["palette"].items():
            setattr(self, k, c(v))
        self.dim = mix(self.text, self.background, 0.4)
        f = t["flame"]
        core, mid, outer = c(f["core"]), c(f["mid"]), c(f["outer"])
        self.flame = [core, mix(core, mid, 0.5), mid, mix(mid, outer, 0.5), outer]
        self.flame_intensity, self.flame_height = f["intensity"], f["height"]
        self.sky_stops = [c(x) for x in t["stratosphere"]["stops"]]
        self.star_density = t["stratosphere"]["star_density"]
        self.edge_glow, self.idle_fill = t["idle_cube"]["edge_glow"], t["idle_cube"]["fill_alpha"]
        self.spin_speed, self.ring = t["working_cube"]["spin_speed"], c(t["working_cube"]["ring"])
        sc, fc = t["still_cube"], t["failed_cube"]
        self.still = (c(sc["color"]), sc["alpha"], sc["edge_style"])
        self.still_fill = c(sc["fill"])
        self.failed = (c(fc["color"]), fc["alpha"], fc["edge_style"])
        r = t["cx_ribbon"]
        self.ribbon = r["enabled"]
        self.ribbon_colors = [c(x) for x in r["colors"]]
        self.ribbon_w, self.ribbon_max, self.ribbon_glow = r["base_width"], r["max_width"], r["glow"]
        rp = t["right_panel"]
        self.panel, self.panel_frac = rp["enabled"], rp["width"]
        self.card_bg, self.card_edge = c(rp["card_bg"]), c(rp["card_edge"])
        sc3 = t.get("scene", vd.DEFAULTS["scene"])
        self.bloom, self.fog = sc3["bloom"], sc3["fog"]
        self.camera_speed, self.exposure = sc3["camera_speed"], sc3["exposure"]

    def gate_edge(self, name):
        return self.gate_cx if name == "cx" else self.gate_x if name in ("x", "sx", "y", "z") else self.gate_1q


LOOK = None  # set by set_look(); Look() of the defaults on first use


def look():
    global LOOK
    if LOOK is None:
        LOOK = Look()
    return LOOK


def set_look(theme, label="theme: default"):
    global LOOK
    LOOK = Look(theme, label)
    return LOOK


# ---------------------------------------------------------------- sources
def _tcp_reader(addr, q):
    host, port = addr.rsplit(":", 1)
    while True:
        try:
            s = socket.create_connection((host, int(port)), timeout=5)
            s.settimeout(None)
            break
        except OSError:
            q.put({"type": "_status", "text": f"waiting for stream on {addr} ..."})
            time.sleep(1)
    q.put({"type": "_status", "text": ""})
    buf = b""
    while True:
        try:
            d = s.recv(65536)
        except OSError:
            d = b""
        if not d:
            q.put({"type": "_status", "text": "stream closed"})
            return
        buf += d
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            ev = parse_line(line)
            if ev:
                q.put(ev)


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        ev = json.loads(line)
    except ValueError:
        return None
    return ev if isinstance(ev, dict) else None


def pace(ev):
    """Seconds to hold after an event during replay/demo so changes are watchable."""
    return {"circuit_update": 1.6, "run_start": 1.2, "equivalence": 0.6, "result": 2.5}.get(ev.get("type"), 0.25)


def _feed(events_iter, q, speed, paused):
    for ev in events_iter:
        while paused.is_set():
            time.sleep(0.05)
        q.put(ev)
        time.sleep(pace(ev) / max(speed, 0.01))
    q.put({"type": "_status", "text": "playback finished"})


def read_jsonl(path):
    with open(path, "rb") as f:
        for line in f:
            ev = parse_line(line)
            if ev:
                yield ev


def layer_gates(gates, nq):
    level = [0] * nq
    out = []
    for name, qs in gates:
        lay = max(level[q] for q in qs)
        for q in qs:
            level[q] = lay + 1
        out.append({"name": name, "qubits": list(qs), "layer": lay})
    return out, (max(level) if level else 0)


def demo_events(seed=3):
    """Synthetic Grover-like run. NOT real data; every event carries demo=True."""
    rng = random.Random(seed)
    nq = 3
    gates = []
    for _ in range(24):
        a, b = rng.sample(range(nq), 2)
        gates += [(rng.choice(["rz", "sx", "x"]), (rng.randrange(nq),)) for _ in range(5)] + [("cx", (a, b))]
    gates += [(rng.choice(["rz", "sx"]), (rng.randrange(nq),)) for _ in range(157 - len(gates))]

    def snap(gs):
        gl, depth = layer_gates(gs, nq)
        return {"gates": len(gs), "depth": depth, "cx": sum(g[0] == "cx" for g in gs), "n_qubits": nq,
                "gate_list": gl, "demo": True}

    yield {"type": "run_start", "circuit": "grover3 (DEMO)", "planner": "demo", "max_calls": 12, **snap(gates)}
    call = 0
    for tgt in [131, 112, 98, 93, 87, 81]:
        for strat in ["level1", "combo"]:
            call += 1
            yield {"type": "tool_call", "call": call, "max_calls": 12, "tool": "run_pass", "strategy": strat, "demo": True}
        single = [i for i, g in enumerate(gates) if g[0] != "cx"]
        drop = set(rng.sample(single, len(gates) - tgt))
        gates = [g for i, g in enumerate(gates) if i not in drop]
        yield {"type": "circuit_update", "circuit_id": f"c{call}", **snap(gates)}
        yield {"type": "equivalence", "ok": True, "demo": True}
    yield {"type": "result", "circuit": "grover3 (DEMO)", "source": "demo", "gates": len(gates), "equiv": True, "demo": True}


# ---------------------------------------------------------------- state
# Gate lifecycle (pure logic, no display needed; see tests/test_events.py):
#   WORKING: the gate is new (or moved to a new layer) in the latest circuit_update, i.e. it was produced
#            by the run_pass / candidate that just landed. It spins and burns with blue flames for
#            WORK_SECS, then settles to IDLE.
#            Working = the turquoise cube frame stays still; the stratosphere INSIDE it spins, ringed by
#            blue flames that stay within the cube's edges.
#   FAILED:  the gate belongs to a candidate circuit whose equivalence check failed / whose finish was
#            rejected. The cube stays on screen (static, no flames) but its inner stratosphere disappears,
#            leaving an empty frame. The next circuit_update (or the restore of the last passing circuit
#            after HOLD_FAIL_SECS) replaces it.
#   REMOVED: the gate was deleted by an optimization step. It is NEVER dropped: it stays on screen,
#            static, dim/grey and empty, frozen at its last layer and pushed slightly behind the wire
#            plane (REMOVED_Z), so the screen is a permanent map of every gate the run has seen. The
#            gate-count ticker still shows only the current circuit's real count. If a later circuit
#            contains the gate again it lights back up.
WORK_SECS, FAIL_SECS, HOLD_FAIL_SECS = 2.5, 0.6, 1.5
IDLE, WORKING, FAILED, REMOVED = "idle", "working", "failed", "removed"


class Gate:
    def __init__(self, key, g, now, fresh):
        self.key, self.name, self.qubits = key, g["name"], list(g["qubits"])
        self.layer = self.target = float(g["layer"])
        self.alpha = 0.0 if fresh else 1.0
        self.work_until = now + WORK_SECS if fresh else -1.0
        self.failed_at = None   # removal time (REMOVED)
        self.eq_failed = False  # equivalence failed (FAILED)
        self.spin = 0.0         # angle of the inner stratosphere

    def state(self, now):
        if self.failed_at is not None:
            return REMOVED
        if self.eq_failed:
            return FAILED
        return WORKING if now < self.work_until else IDLE

    def fail(self, now):
        """Removed by an optimization step: dissolve."""
        if self.failed_at is None:
            self.failed_at = now

    def dissolve(self, now):
        """Kept for compatibility: cubes never dissolve any more (always 0)."""
        return 0.0


def keyed(gate_list):
    """Stable identity per gate: (name, qubits, k-th occurrence) so unchanged gates persist across updates."""
    seen, out = {}, []
    for g in gate_list:
        k = (g["name"], tuple(g["qubits"]))
        seen[k] = seen.get(k, 0) + 1
        out.append((k + (seen[k],), g))
    return out


class State:
    def __init__(self):
        self.gates, self.nq, self.depth, self.view_depth = {}, 1, 1, 1.0
        self.before = self.now_gates = self.cx = self.depth0 = self.cx0 = None
        self.shown = 0.0
        self.step, self.circuit, self.equiv, self.status = "", "", None, ""
        self.demo, self.done, self.flash = False, False, -99
        self.snaps, self.cur_id, self.last_good, self.restore_at = {}, None, None, None

    def apply(self, ev, now):
        self.version = getattr(self, "version", 0) + 1  # lets the 3D view skip instance rebuilds
        t = ev.get("type")
        if ev.get("demo"):
            self.demo = True
        if t == "_status":
            self.status = ev["text"]
        elif t == "run_start":
            self.gates.clear()
            self.snaps = {"c0": ev}
            self.cur_id = self.last_good_id = "c0"
            self.last_good, self.restore_at = ev, None
            self.circuit = ev.get("circuit", "")
            self.before, self.depth0, self.cx0 = ev.get("gates"), ev.get("depth"), ev.get("cx")
            self.shown = float(ev.get("gates") or 0)
            self.equiv, self.done, self.step = None, False, f"start - {ev.get('planner', '')} planner"
            self._circuit(ev, now, fresh=False)
        elif t == "tool_call":
            if ev.get("round") is not None:
                self.step = f"round {ev['round']} - {ev.get('tool')} {ev.get('strategy') or ''}".strip()
            else:
                self.step = f"call {ev.get('call')}/{ev.get('max_calls')} - {ev.get('tool')} {ev.get('strategy') or ''}".strip()
        elif t == "circuit_update":
            cid = ev.get("circuit_id")
            if cid:
                self.snaps[cid] = ev
            self.cur_id = cid
            self._circuit(ev, now, fresh=True)
        elif t == "equivalence":
            ok = bool(ev.get("ok"))
            self.equiv = ok
            cid = ev.get("circuit_id") or self.cur_id
            if ok:
                if cid in self.snaps:
                    self.last_good = self.snaps[cid]
            else:
                self.fail_circuit(now)
        elif t == "result":
            self.done = True
            if ev.get("equiv") is not None:
                self.equiv = bool(ev["equiv"])
            self.step = f"done - {ev.get('source')} - {ev.get('strategy') or ''}".strip(" -")

    def fail_circuit(self, now):
        """Every cube of the failed candidate goes static and empty; later the last passing circuit returns."""
        for g in self.gates.values():
            if g.failed_at is None:
                g.eq_failed, g.work_until = True, -1.0
        self.restore_at = now + HOLD_FAIL_SECS

    def _circuit(self, ev, now, fresh):
        gl = ev.get("gate_list") or []
        self.nq = max(1, ev.get("n_qubits") or 1 + max((q for g in gl for q in g["qubits"]), default=0))
        self.now_gates, self.depth, self.cx = ev.get("gates"), max(1, ev.get("depth") or 1), ev.get("cx")
        new = dict(keyed(gl))
        changed = False
        for k, g in list(self.gates.items()):
            if k in new and g.failed_at is None:
                if g.eq_failed and fresh:  # back in a new circuit: alive again
                    g.eq_failed, g.work_until = False, now + WORK_SECS
                lay = float(new[k]["layer"])
                if fresh and abs(lay - g.target) > 0.5:  # moved by the step: it is working too
                    g.work_until = now + WORK_SECS
                g.target = lay
            elif k in new:  # removed earlier, back in the new circuit: light it up again
                g.failed_at, g.eq_failed = None, False
                g.target, g.work_until = float(new[k]["layer"]), (now + WORK_SECS if fresh else -1.0)
            elif g.failed_at is None:
                g.fail(now)
                changed = True
        for k, g in new.items():
            if k not in self.gates:
                self.gates[k] = Gate(k, g, now, fresh)
                changed = True
        if changed and fresh:
            self.flash = now

    def tick(self, dt, now):
        if self.restore_at is not None and now >= self.restore_at:
            self.restore_at = None
            if self.last_good is not None:
                self._circuit(self.last_good, now, fresh=True)
        for k in list(self.gates):
            g = self.gates[k]
            st = g.state(now)
            if st != REMOVED:
                g.layer += (g.target - g.layer) * min(1, dt * 3)
            if st == REMOVED:
                continue  # frozen in place: no repack, no spin
            g.alpha = min(1.0, g.alpha + dt * 1.5)
            if st == WORKING:
                g.spin += dt * 2.5 * look().spin_speed
            elif st == IDLE:  # settle: ease back to a square pose
                g.spin = round(g.spin / (math.pi / 2)) * (math.pi / 2) * min(1, dt * 4) + g.spin * (1 - min(1, dt * 4))
        self.view_depth += (self.depth - self.view_depth) * min(1, dt * 2)
        if self.now_gates is not None:
            d = self.now_gates - self.shown
            self.shown += d * min(1, dt * 2.5) if abs(d) > 0.5 else d

    def working(self, now):
        return [g for g in self.gates.values() if g.state(now) == WORKING]

    def visible(self, now):
        """Gates drawn this frame: all of them (the permanent circuit map)."""
        return list(self.gates.values())

    def current(self, now):
        return [g for g in self.gates.values() if g.state(now) != REMOVED]


# ---------------------------------------------------------------- render
def make_sky(w, h, seed=7, stops=None, star_density=None):
    """Procedural stratosphere texture (drawn INSIDE each cube): zenith -> mid -> low gradient
    (theme stops 0-2), thin glowing horizon band (stop 3), dark limb, soft haze, tiny stars."""
    L = look()
    stops = stops or L.sky_stops
    star_density = L.star_density if star_density is None else star_density
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h)[None, :, None]
    top, mid, low = (np.array(stops[i], float) for i in range(3))
    band = np.array(stops[-1], float)
    col = np.where(y < 0.6, top + (mid - top) * (y / 0.6), mid + (low - mid) * ((y - 0.6) / 0.4))
    col = col + np.exp(-((y - 0.72) / 0.04) ** 2) * band * 0.9
    col = col + np.exp(-((y - 0.74) / 0.14) ** 2) * np.array([70, 40, 60])
    col = col + (y > 0.76) * (y - 0.76) * np.array([-220, -160, -190])
    img = np.broadcast_to(col, (w, h, 3)).copy()
    small = rng.normal(0, 1, (w // 16 + 2, h // 16 + 2))
    xs, ys = np.linspace(0, small.shape[0] - 1, w), np.linspace(0, small.shape[1] - 1, h)
    small = np.array([np.interp(ys, np.arange(small.shape[1]), row) for row in small])
    haze = np.array([np.interp(xs, np.arange(small.shape[0]), c) for c in small.T]).T[:, :, None]
    img += haze * 8 * np.exp(-((y - 0.65) / 0.2) ** 2)
    n = int(w * h // 120 * star_density)
    sx, sy = rng.integers(0, w, n), rng.integers(0, int(h * 0.68), n)
    img[sx, sy] += (rng.random(n) ** 3 * 230)[:, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def make_background(w, h, seed=11):
    """Plain near-black void with very faint stars (the sky lives inside the cubes now)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((w, h, 3)) + np.array(look().background, float)
    n = w * h // 2500
    img[rng.integers(0, w, n), rng.integers(0, h, n)] += (rng.random(n) ** 4 * 70)[:, None]
    return np.clip(img, 0, 255).astype(np.uint8)


CUBE_V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
NORMALS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], float)
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def spin_matrix(a):
    c, s = math.cos(a), math.sin(a)
    tilt = 0.35 * math.sin(a * 0.5)
    ct, st = math.cos(tilt), math.sin(tilt)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])


class Camera:
    def __init__(self):
        self.yaw = self.pitch = 0.0
        self.dist, self.fov = 20.0, 1000

    def rot(self):
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return rx @ ry

    def project(self, pts, R):
        c = pts @ R.T
        z = np.maximum(c[:, 2] + self.dist, 0.5)
        f = self.fov / z
        return np.stack([scene_w() / 2 + c[:, 0] * f, H * 0.46 - c[:, 1] * f], 1), z


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Renderer:
    def __init__(self, pg):
        self.pg = pg
        self.cache, self.flames, self.rng = {}, [], random.Random(5)
        self.set_theme()

    def set_theme(self):
        """(Re)build textures from the current look (called when the director's theme arrives)."""
        self.sky = self.pg.surfarray.make_surface(make_sky(128, 128))
        self.bg = self.pg.surfarray.make_surface(make_background(W, H))
        self.cache.clear()

    def resize(self):
        self.bg = self.pg.surfarray.make_surface(make_background(W, H))

    def sky_at(self, w, h):
        k = (w, h)
        if k not in self.cache:
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[k] = self.pg.transform.smoothscale(self.sky, k).convert_alpha()
        return self.cache[k]

    def sky_fill(self, glow, face_polys, alpha, scroll):
        """Clip the pre-rendered stratosphere to the cube's visible faces (its silhouette)."""
        pg = self.pg
        pts = [p for f in face_polys for p in f]
        x0, y0 = int(min(p[0] for p in pts)), int(min(p[1] for p in pts))
        x1, y1 = int(max(p[0] for p in pts)) + 1, int(max(p[1] for p in pts)) + 1
        w, h = x1 - x0, y1 - y0
        if w < 2 or h < 2 or w > W or h > H:
            return
        tile = pg.Surface((w, h), pg.SRCALPHA)
        sky = self.sky_at(w * 2, h)
        tile.blit(sky, (-int(scroll % 1.0 * w), 0))
        mask = pg.Surface((w, h), pg.SRCALPHA)
        for f in face_polys:
            pg.draw.polygon(mask, (255, 255, 255, int(235 * alpha)), [(p[0] - x0, p[1] - y0) for p in f])
        tile.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        glow.blit(tile, (x0, y0))

    def inner_sky(self, glow, cxy, r, angle, alpha):
        """Spinning disc of stratosphere inside the cube."""
        pg = self.pg
        d = max(4, int(r * 2))
        tile = pg.transform.rotate(self.sky_at(d, d), math.degrees(angle))
        w, h = tile.get_size()
        mask = pg.Surface((w, h), pg.SRCALPHA)
        pg.draw.circle(mask, (255, 255, 255, int(245 * alpha)), (w / 2, h / 2), r)
        tile = tile.copy()
        tile.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        glow.blit(tile, (cxy[0] - w / 2, cxy[1] - h / 2))

    def emit_flames(self, cx, cy, r_in, r_max, dt):
        """Blue flame particles born on the ring around the inner sky; clipped to the cube (r_max)."""
        L = look()
        for _ in range(max(1, int(90 * L.flame_intensity * dt * max(1, r_in / 10)))):
            a = self.rng.uniform(0, 2 * math.pi)
            self.flames.append([cx + math.cos(a) * r_in, cy + math.sin(a) * r_in,
                                self.rng.uniform(-6, 6),
                                -self.rng.uniform(10, 30) * max(0.5, r_in / 14) * L.flame_height,
                                0.0, self.rng.uniform(0.25, 0.5), self.rng.uniform(1.0, 2.4) * max(0.6, r_in / 14),
                                cx, cy, r_max])

    def draw_flames(self, glow, dt):
        pg, alive, FLAME = self.pg, [], look().flame
        for f in self.flames:
            f[4] += dt
            if f[4] >= f[5]:
                continue
            f[0] += f[2] * dt + math.sin(f[4] * 25 + f[1]) * 0.4
            f[1] += f[3] * dt
            if math.hypot(f[0] - f[7], f[1] - f[8]) > f[9]:
                continue  # flames stay inside the cube
            u = f[4] / f[5]
            c = FLAME[min(len(FLAME) - 1, int(u * len(FLAME)))]
            r = f[6] * (1 - u * 0.6)
            pg.draw.circle(glow, (*c, int(55 * (1 - u))), (f[0], f[1]), r * 2.2)
            pg.draw.circle(glow, (*c, int(220 * (1 - u))), (f[0], f[1]), max(1, r))
            alive.append(f)
        self.flames = alive[-4000:]

    def ribbon(self, glow, a, b, activity, al):
        """CX link as a glowing gradient ribbon; width grows with activity (0..1)."""
        L, pg = look(), self.pg
        w = L.ribbon_w + (L.ribbon_max - L.ribbon_w) * activity
        cols, n = L.ribbon_colors, 12
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            p0 = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0)
            p1 = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)
            u = (t0 + t1) / 2 * (len(cols) - 1)
            c = mix(cols[int(u)], cols[min(len(cols) - 1, int(u) + 1)], u - int(u))
            pg.draw.line(glow, (*c, int(70 * L.ribbon_glow * al)), p0, p1, int(w * 2))
            pg.draw.line(glow, (*c, int(235 * al)), p0, p1, max(1, int(w)))

    def edges(self, glow, vp, color, alpha, style, glow_k=1.0):
        pg = self.pg
        for a_, b_ in EDGES:
            p, q = vp[a_], vp[b_]
            if style == "solid":
                pg.draw.line(glow, (*color, int(50 * alpha * glow_k)), p, q, 3)
                pg.draw.aaline(glow, (*mix(color, WHITE, 0.35), int(245 * alpha)), p, q)
                continue
            n = 6 if style == "dashed" else 10
            for i in range(n):
                if style == "dashed" and i % 2:
                    continue
                t0 = i / n
                t1 = t0 + (1 / n if style == "dashed" else 0.03)
                pa = (p[0] + (q[0] - p[0]) * t0, p[1] + (q[1] - p[1]) * t0)
                pb = (p[0] + (q[0] - p[0]) * t1, p[1] + (q[1] - p[1]) * t1)
                if style == "dotted":
                    pg.draw.circle(glow, (*color, int(245 * alpha)), pa, 1.5)
                else:
                    pg.draw.line(glow, (*color, int(245 * alpha)), pa, pb, 2)

    def draw(self, screen, st, cam, fonts, now, dt):
        pg, L = self.pg, look()
        screen.blit(self.bg, (0, 0))
        glow = pg.Surface((W, H), pg.SRCALPHA)
        nq, depth = st.nq, max(st.view_depth, 1)
        sp, gap = 1.0, 2.6
        span = depth * sp
        cam.dist = max(span * 0.62, nq * gap * 2.2, 10)
        cam.yaw = 0.22 * math.sin(now * 0.07)
        cam.pitch = 0.22 + 0.06 * math.sin(now * 0.11)
        R = cam.rot()

        def wy(q):
            return ((nq - 1) / 2 - q) * gap

        for q in range(nq):
            p, _ = cam.project(np.array([[-span / 2 - 3, wy(q), 0], [span / 2 + 3, wy(q), 0]]), R)
            pg.draw.line(glow, (*L.wire, 30), p[0], p[1], 5)
            pg.draw.line(glow, (*L.wire, 140), p[0], p[1], 1)
        cubes, links = [], []
        for g in st.gates.values():
            x = (g.layer - depth / 2 + 0.5) * sp
            state = g.state(now)
            d = g.dissolve(now)
            vis = g.alpha * (1 - d)
            if vis <= 0.01:
                continue
            s = 0.4 * sp * (0.6 + 0.4 * g.alpha) * (1.25 if g.name == "cx" else 1) * (1 - 0.6 * d)
            z = REMOVED_Z if state == REMOVED else 0.0
            for i, q in enumerate(g.qubits):
                cubes.append((np.array([x, wy(q), z]), s * (0.8 if g.name == "cx" and i == 0 else 1), g, state, vis, d))
            if len(g.qubits) == 2:
                links.append((np.array([[x, wy(g.qubits[0]), z], [x, wy(g.qubits[1]), z]]), vis, state))
        for pts, al, state in links:
            p, _ = cam.project(pts, R)
            if state in (REMOVED, FAILED):
                c, a2, _ = L.still if state == REMOVED else L.failed
                pg.draw.line(glow, (*c, int(200 * a2 * al)), p[0], p[1], 1)
            elif L.ribbon:
                self.ribbon(glow, p[0], p[1], 1.0 if state == WORKING else 0.15, al)
            else:
                pg.draw.line(glow, (*L.cx_link, int(70 * al)), p[0], p[1], 7)
                pg.draw.line(glow, (*L.cx_link, int(220 * al)), p[0], p[1], 2)
        if cubes:
            _, zc = cam.project(np.array([c[0] for c in cubes]), R)
            nr = NORMALS @ R.T
            for i in np.argsort(-zc):
                ctr, s, g, state, vis, d = cubes[i]
                vp, vz = cam.project(ctr + CUBE_V * s, R)  # the frame itself never spins
                faces = [[vp[k] for k in f] for fi, f in enumerate(FACES) if nr[fi, 2] < 0]
                cxy = vp.mean(0)
                rad = s * cam.fov / vz.mean()  # ~ half the cube's on-screen size
                if state == WORKING:
                    for f in faces:  # dark glass so the spinning sky + flames read clearly
                        pg.draw.polygon(glow, (*L.background, int(200 * vis)), f)
                    self.inner_sky(glow, cxy, rad * 0.55, g.spin, vis)
                    pg.draw.circle(glow, (*L.ring, int(120 * vis)), cxy, rad * 0.6, 2)
                    self.emit_flames(cxy[0], cxy[1], rad * 0.6, rad * 0.95, dt)
                    self.edges(glow, vp, L.gate_edge(g.name), vis, "solid", L.edge_glow * 1.6)
                elif state == IDLE:
                    self.sky_fill(glow, faces, vis * max(0.25, L.idle_fill), 0.0)
                    self.edges(glow, vp, L.gate_edge(g.name), vis, "solid", L.edge_glow)
                else:  # FAILED / REMOVED: empty, static frame in the theme's 'still' style
                    color, a2, style = L.still if state == REMOVED else L.failed
                    for f in faces:
                        pg.draw.polygon(glow, (*L.still_fill, int(170 * vis * a2)), f)
                    self.edges(glow, vp, color, vis * a2, style, 0.5)
        self.draw_flames(glow, dt)
        screen.blit(glow, (0, 0))
        if L.panel:
            draw_panel(screen, st, fonts, now, pg)
        draw_overlay(screen, st, fonts, now, pg)
        draw_frame(screen, pg)


TURQUOISE = (64, 224, 208)  # #40E0D0, the default window frame (theme palette.window_frame)
REMOVED_Z = 1.4


def scene_w():
    """Width of the 3D scene area (narrower when the theme enables the right panel)."""
    L = look()
    return int(W * (1 - L.panel_frac)) if L.panel else W


def draw_panel(screen, st, fonts, now, pg):
    """Slim right panel of big-number cards (gates, depth, CX, equiv), dashboard-style."""
    L, (big, mono, sans, small) = look(), fonts
    x0, pw = scene_w(), W - scene_w()
    panel = pg.Surface((pw, H), pg.SRCALPHA)
    panel.fill((*L.background, 235))
    screen.blit(panel, (x0, 0))
    pg.draw.line(screen, L.card_edge, (x0, 0), (x0, H), 1)
    eq = None if st.equiv is None else ("PASS" if st.equiv else "FAIL")
    cards = [("GATES", None if st.now_gates is None else int(round(st.shown)), st.before, L.accent if now - st.flash < 1.2 else L.text),
             ("DEPTH", st.depth if st.now_gates is not None else None, st.depth0, L.text),
             ("CX", st.cx, st.cx0, L.text),
             ("EQUIV", eq, None, L.badge_pass if st.equiv else L.badge_fail if st.equiv is not None else L.dim)]
    pad, top = 14, 28
    ch = (H - top - 40 - pad * 3) // 4
    num = pg.font.SysFont("dejavusansmono,menlo,monospace", max(24, min(64, int(pw * 0.26), ch // 2)), bold=True)
    for i, (name, val, was, col) in enumerate(cards):
        r = pg.Rect(x0 + pad, top + i * (ch + pad), pw - 2 * pad, ch)
        pg.draw.rect(screen, L.card_bg, r, border_radius=10)
        pg.draw.rect(screen, L.card_edge if name != "EQUIV" or val is None else col, r, 1, border_radius=10)
        screen.blit(small.render(name, True, L.dim), (r.x + 12, r.y + 8))
        t = num.render("-" if val is None else str(val), True, col)
        screen.blit(t, (r.x + 12, r.centery - t.get_height() / 2 + 6))
        if was is not None:
            w_ = small.render(f"from {was}", True, L.dim)
            screen.blit(w_, (r.right - w_.get_width() - 10, r.bottom - w_.get_height() - 6))


def draw_frame(screen, pg, width=4):
    """Window frame (theme palette.window_frame, default turquoise #40E0D0) with a subtle inner glow."""
    TURQUOISE = look().window_frame
    glow = pg.Surface((W, H), pg.SRCALPHA)
    for i, al in enumerate((60, 35, 18, 8)):
        pg.draw.rect(glow, (*TURQUOISE, al), (width + i * 3, width + i * 3, W - 2 * (width + i * 3),
                                              H - 2 * (width + i * 3)), 3)
    screen.blit(glow, (0, 0))
    pg.draw.rect(screen, TURQUOISE, (0, 0, W, H), width)


def draw_overlay(screen, st, fonts, now, pg):
    big, mono, sans, small = fonts
    L = look()
    AMBER, WHITE, DIM, GREEN, RED = L.accent, L.text, L.dim, L.badge_pass, L.badge_fail
    lab = small.render(L.label, True, mix(L.dim, L.background, 0.35))
    screen.blit(lab, (14, 10))
    if L.panel:  # numbers + equiv live in the right panel; keep only step/status/name here
        sw = scene_w()
        if st.step:
            s = mono.render(st.step, True, WHITE)
            screen.blit(s, (sw / 2 - s.get_width() / 2, H - 56))
        if st.status:
            s = small.render(st.status, True, DIM)
            screen.blit(s, (sw / 2 - s.get_width() / 2, H - 26))
        if st.circuit:
            screen.blit(sans.render(st.circuit, True, DIM), (40, 36))
        if st.demo:
            screen.blit(sans.render("DEMO - synthetic data", True, AMBER), (40, 62))
        return
    x0, y0 = 40, 30
    t = big.render(f"{int(round(st.shown))}", True, AMBER if now - st.flash < 1.2 else WHITE)
    screen.blit(t, (x0, y0))
    lbl = sans.render("gates", True, DIM)
    screen.blit(lbl, (x0 + t.get_width() + 14, y0 + t.get_height() - lbl.get_height() - 12))
    b = mono.render(f"from {st.before if st.before is not None else '-'}", True, DIM)
    screen.blit(b, (x0 + 4, y0 + t.get_height()))
    y = y0 + t.get_height() + b.get_height() + 10
    for name, cur, was in (("depth", st.depth if st.now_gates is not None else None, st.depth0), ("CX", st.cx, st.cx0)):
        screen.blit(sans.render(name, True, DIM), (x0 + 4, y))
        screen.blit(mono.render(f"{'-' if was is None else was} -> {'-' if cur is None else cur}", True, WHITE), (x0 + 90, y))
        y += 34
    if st.step:
        s = mono.render(st.step, True, WHITE)
        screen.blit(s, (W / 2 - s.get_width() / 2, H - 56))
    if st.status:
        s = small.render(st.status, True, DIM)
        screen.blit(s, (W / 2 - s.get_width() / 2, H - 26))
    if st.equiv is not None:
        col = GREEN if st.equiv else RED
        txt = sans.render("EQUIV PASS" if st.equiv else "EQUIV FAIL", True, L.background)
        r = txt.get_rect(topright=(W - 40, 36)).inflate(28, 14)
        halo = pg.Surface((r.w + 30, r.h + 30), pg.SRCALPHA)
        pg.draw.rect(halo, (*col, 50), halo.get_rect(), border_radius=30)
        screen.blit(halo, (r.x - 15, r.y - 15))
        pg.draw.rect(screen, col, r, border_radius=20)
        screen.blit(txt, txt.get_rect(center=r.center))
    if st.circuit:
        c = small.render(st.circuit, True, DIM)
        screen.blit(c, (W - 40 - c.get_width(), 96))
    if st.demo:
        d = sans.render("DEMO - synthetic data", True, AMBER)
        screen.blit(d, (W - 40 - d.get_width(), H - 56))


def fonts(pg):
    sans = "dejavusans,helvetica,arial,liberationsans"
    mono = "dejavusansmono,menlo,sfmono,liberationmono,monospace"
    return (pg.font.SysFont(mono, 96, bold=True), pg.font.SysFont(mono, 26),
            pg.font.SysFont(sans, 24), pg.font.SysFont(sans, 18))


def make_source(a):
    """Start the event source thread (live TCP / replay / demo); returns (queue, paused)."""
    q, paused = queue.Queue(), threading.Event()
    if a.connect:
        th = threading.Thread(target=_tcp_reader, args=(a.connect, q), daemon=True)
    else:
        it = demo_events() if a.demo else read_jsonl(a.replay)
        th = threading.Thread(target=_feed, args=(it, q, a.speed, paused), daemon=True)
    th.start()
    return q, paused


def main(argv=None):
    global W, H
    p = argparse.ArgumentParser(description="Legionary live 3D viewer")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--connect", metavar="HOST:PORT", help="live TCP stream from orchestrator --events tcp:PORT")
    src.add_argument("--replay", metavar="FILE.jsonl", help="play back a saved event file")
    src.add_argument("--demo", action="store_true", help="synthetic Grover-like run (not real data)")
    p.add_argument("--speed", type=float, default=1.0, help="replay/demo speed multiplier")
    p.add_argument("--screenshot", metavar="PNG", help="save a frame to PNG and exit (headless testing)")
    p.add_argument("--screenshot-at", type=float, default=6.0, help="seconds after start for --screenshot")
    p.add_argument("--size", default=f"{W}x{H}")
    p.add_argument("--director-backend", default="auto", choices=["auto", "meta", "fireworks", "nebius", "off"],
                   help="Muse Spark visual director backend (asked once at startup; off = built-in theme)")
    p.add_argument("--director-refresh", action="store_true", help="ask the model even if a cached theme exists")
    p.add_argument("--renderer", choices=["3d", "2d"], default="3d",
                   help="3d: cinematic GPU scene (moderngl, OpenGL 3.3 core); 2d: the original pygame renderer")
    p.add_argument("--angle", type=float, default=0.0, help="3d: extra camera orbit angle in degrees")
    p.add_argument("--render-scale", type=float, default=0.0,
                   help="3d: internal scene resolution scale 0.25-1 (default auto: 0.5 on Retina, else 0.75)")
    p.add_argument("--stats", action="store_true", help="3d: show fps / draw calls / instance counts")
    p.add_argument("--director-wait", type=float, default=0.0,
                   help="seconds to wait for the model theme before drawing (screenshots)")
    p.add_argument("--nemotron-review", action=argparse.BooleanOptionalAction, default=None,
                   help="accepted for parity with the orchestrator; the viewer never runs Nemotron")
    a = p.parse_args(argv)
    if a.render_scale:
        a.render_scale = min(1.0, max(0.25, a.render_scale))
    W, H = map(int, a.size.split("x"))
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame as pg
    if a.renderer == "3d":
        director = vd.Director(a.director_backend, a.director_refresh).start()
        if a.director_wait and director.thread:
            director.thread.join(a.director_wait)
        director.poll()
        set_look(director.theme, director.label)
        from . import gl3d
        return gl3d.run(a, director)
    pg.init()
    screen = pg.display.set_mode((W, H), 0 if a.screenshot else pg.RESIZABLE)
    pg.display.set_caption("Legionary - live circuit")
    director = vd.Director(a.director_backend, a.director_refresh).start()
    if a.director_wait and director.thread:
        director.thread.join(a.director_wait)
    director.poll()
    set_look(director.theme, director.label)
    rend = Renderer(pg)
    q, paused = queue.Queue(), threading.Event()
    if a.connect:
        th = threading.Thread(target=_tcp_reader, args=(a.connect, q), daemon=True)
    else:
        it = demo_events() if a.demo else read_jsonl(a.replay)
        th = threading.Thread(target=_feed, args=(it, q, a.speed, paused), daemon=True)
    th.start()
    st, cam, fs, clock = State(), Camera(), fonts(pg), pg.time.Clock()
    t0 = time.time()
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_ESCAPE, pg.K_q)):
                pg.quit(); return 0
            if e.type == pg.KEYDOWN and e.key == pg.K_SPACE:
                paused.clear() if paused.is_set() else paused.set()
            if e.type == pg.VIDEORESIZE:
                W, H = e.w, e.h
                screen = pg.display.set_mode((W, H), pg.RESIZABLE)
                rend.resize()
        now = time.time() - t0
        if director.poll():  # model theme arrived: swap it in
            set_look(director.theme, director.label)
            rend.set_theme()
        while not q.empty():
            st.apply(q.get(), now)
        dt = clock.get_time() / 1000 or 1 / 60
        st.tick(dt, now)
        rend.draw(screen, st, cam, fs, now, dt)
        pg.display.flip()
        clock.tick(60)
        if a.screenshot and now >= a.screenshot_at:
            pg.image.save(screen, a.screenshot)
            print(f"saved {a.screenshot}")
            pg.quit(); return 0


if __name__ == "__main__":
    sys.exit(main())
