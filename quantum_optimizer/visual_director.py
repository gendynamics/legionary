"""Muse Spark as the viewer's 'visual director'.

Asked ONCE at viewer startup (never per frame), in a background thread, for a JSON theme that
follows THEME_SCHEMA. The reply is validated (hex colors, clamped numbers), merged onto DEFAULTS,
and cached (~/.cache/legionary/theme.json, with model + timestamp). Any error/timeout falls back to
the cache, then to DEFAULTS. Keys are read from the environment only (agent_planner.PROVIDERS)."""
import base64, copy, json, os, re, threading, time

CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "legionary", "theme.json")
TIMEOUT = 45.0  # real nebius call took 20.2 s
_REF = "/cursor/stores/store-1cbd7f0e-8c9f-44c5-b042-777325a1ec82/grok-bot/agents/169c6326-7fa5-463f-a12d-97d693dc75ae/attachments/"
REFERENCE_IMAGES = [_REF + "01M69XYC4K6M97QXD67MJAEGWP.jpg", _REF + "01M69XYC8ZPN4VQAZHTJM2C4NT.jpg"]

# Built-in defaults == the hand-made v3 look. The only hardcoded palette in the viewer.
DEFAULTS = {
    "palette": {"background": "#04040c", "window_frame": "#40e0d0", "wire": "#b4e6ff",
                "gate_1q": "#40e0d0", "gate_x": "#40e0d0", "gate_cx": "#8fc9e8", "cx_link": "#eb5adc",
                "accent": "#ffb032", "text": "#ebf0ff", "badge_pass": "#50eb8c", "badge_fail": "#ff4650"},
    "flame": {"core": "#ffffff", "mid": "#40e0d0", "outer": "#0b6b74", "intensity": 1.0, "height": 1.0},
    "stratosphere": {"stops": ["#06081e", "#181650", "#46307d", "#78a0d7"], "star_density": 1.0},
    "idle_cube": {"edge_glow": 1.0, "fill_alpha": 0.92},
    "working_cube": {"spin_speed": 1.0, "ring": "#50aaff"},
    "still_cube": {"color": "#6e7882", "alpha": 0.55, "edge_style": "dashed", "fill": "#06060a"},
    "failed_cube": {"color": "#e0505a", "alpha": 0.8, "edge_style": "dotted"},
    "cx_ribbon": {"enabled": False, "colors": ["#46e1ff", "#aa6eff", "#eb5adc"], "base_width": 2.0,
                  "max_width": 8.0, "glow": 0.5},
    "right_panel": {"enabled": False, "card_bg": "#0c1020", "card_edge": "#2a3a5a", "width": 0.17},
    "scene": {"bloom": 1.0, "fog": 0.018, "camera_speed": 1.0, "exposure": 1.1},
    "rationale": "",
}
RANGES = {("flame", "intensity"): (0.2, 2.0), ("flame", "height"): (0.3, 2.0),
          ("stratosphere", "star_density"): (0.0, 3.0), ("idle_cube", "edge_glow"): (0.2, 2.0),
          ("idle_cube", "fill_alpha"): (0.1, 1.0), ("working_cube", "spin_speed"): (0.2, 3.0),
          ("still_cube", "alpha"): (0.2, 0.9), ("failed_cube", "alpha"): (0.2, 0.9),
          ("cx_ribbon", "base_width"): (1.0, 6.0), ("cx_ribbon", "max_width"): (2.0, 16.0),
          ("cx_ribbon", "glow"): (0.0, 1.0), ("right_panel", "width"): (0.12, 0.24),
          ("scene", "bloom"): (0.0, 2.5), ("scene", "fog"): (0.0, 0.08),
          ("scene", "camera_speed"): (0.2, 2.0), ("scene", "exposure"): (0.5, 2.0)}
EDGE_STYLES = ("solid", "dashed", "dotted")
HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _schema():
    col = {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"}
    num = {"type": "number"}

    def obj(d):
        props = {k: (col if isinstance(v, str) and v.startswith("#") else
                     {"type": "array", "items": col} if isinstance(v, list) else
                     {"type": "boolean"} if isinstance(v, bool) else
                     {"type": "string", "enum": list(EDGE_STYLES)} if k == "edge_style" else
                     {"type": "string"} if isinstance(v, str) else num) for k, v in d.items()}
        return {"type": "object", "properties": props, "required": list(d), "additionalProperties": False}
    props = {k: obj(v) for k, v in DEFAULTS.items() if isinstance(v, dict)}
    props["rationale"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": list(DEFAULTS), "additionalProperties": False}


THEME_SCHEMA = _schema()

PROMPT = """You are the visual director for "Legionary", a native pygame 3D viewer that shows a quantum
circuit being optimized live (hackathon demo; must earn a standing ovation, and be ADHD-friendly:
glanceable, calm, one clear focal point, no clutter).
Reference image 1 (cubes): a grid of see-through wireframe cubes floating in black space, thin luminous
edges, each cube filled with glowing gold and lavender points like a star map.
Reference image 2 (dashboard): dark-mode security dashboard, three-column flow, glowing cyan->violet->
magenta ribbons that thicken with traffic, slim right panel with big numbers on small cards.
It is rendered as a cinematic GPU 3D scene (NVIDIA-keynote scale): low perspective camera over a vast
field of cubes receding into darkness, additive bloom, depth fog, slow dolly/orbit.
Fixed constraints of the viewer: near-black background; a window frame border (currently turquoise
#40E0D0, keep it turquoise-family); each gate is a cube holding a stratosphere gradient (zenith ->
horizon glow) inside; cubes being worked on spin the inner stratosphere with a ring of TURQUOISE fire flames (white core ->
turquoise/cyan -> deep teal) inside the cube;
removed gates and failed-equivalence gates stay on the map as STILL frames that must read clearly as
'stayed/removed' vs 'failed'; two-qubit CX gates are linked by a line (optionally a glowing ribbon
that thickens with activity); a right panel of big-number cards (gates, depth, CX, equiv) is optional.
Choose the scene bloom (0-2.5), fog density (0-0.08), camera speed (0.2-2) and exposure (0.5-2), the
palette, flame colors, stratosphere stops (4 hex, top to horizon), star density, cube
styles, still/failed cube styles, ribbon and panel. Text must stay readable on the background.
Reply with ONLY one JSON object matching this JSON schema (all hex colors '#rrggbb'):
""" + json.dumps(THEME_SCHEMA, separators=(",", ":"))


def hex_rgb(s):
    m = HEX.match(str(s).strip())
    if not m:
        raise ValueError(f"bad color {s!r}")
    v = m.group(1)
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def _norm_hex(s):
    return "#%02x%02x%02x" % hex_rgb(s)


def validate(raw):
    """Merge a model reply onto DEFAULTS: invalid colors/types are dropped (default kept), numbers
    clamped to RANGES, unknown keys ignored. Returns (theme, warnings)."""
    theme, warn = copy.deepcopy(DEFAULTS), []
    if not isinstance(raw, dict):
        return theme, ["theme is not an object"]
    for sec, dv in DEFAULTS.items():
        if sec not in raw:
            continue
        if not isinstance(dv, dict):
            if isinstance(raw[sec], str):
                theme[sec] = raw[sec][:300]
            continue
        rv = raw[sec]
        if not isinstance(rv, dict):
            warn.append(f"{sec}: not an object"); continue
        for k, d in dv.items():
            if k not in rv:
                continue
            v = rv[k]
            try:
                if isinstance(d, bool):
                    if not isinstance(v, bool):
                        raise ValueError("not bool")
                elif isinstance(d, str) and d.startswith("#"):
                    v = _norm_hex(v)
                elif isinstance(d, list):
                    v = [_norm_hex(x) for x in v][:6]
                    if len(v) < 2:
                        raise ValueError("need >=2 colors")
                elif k == "edge_style":
                    if v not in EDGE_STYLES:
                        raise ValueError(v)
                elif isinstance(d, (int, float)):
                    if isinstance(v, bool):
                        raise ValueError("bool")
                    v = float(v)
                    if v != v:
                        raise ValueError("nan")
                    lo, hi = RANGES.get((sec, k), (0.0, 1e9))
                    if not lo <= v <= hi:
                        warn.append(f"{sec}.{k}={v} clamped")
                    v = min(hi, max(lo, v))
                theme[sec][k] = v
            except (ValueError, TypeError) as e:
                warn.append(f"{sec}.{k}: {e}")
    if theme["cx_ribbon"]["max_width"] < theme["cx_ribbon"]["base_width"]:
        theme["cx_ribbon"]["max_width"] = theme["cx_ribbon"]["base_width"]
    return theme, warn


def extract_json(text):
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        raise ValueError("no JSON object in reply")
    return json.loads(text[i:j + 1])


def load_cache(path=CACHE_PATH):
    try:
        with open(path) as f:
            d = json.load(f)
        theme, _ = validate(d.get("theme", {}))
        return theme, d
    except Exception:
        return None, None


def save_cache(theme, meta, path=CACHE_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({**meta, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "theme": theme}, f, indent=2)


def _images():
    out = []
    for p in REFERENCE_IMAGES:
        if os.path.exists(p):
            with open(p, "rb") as f:
                out.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode())
    return out


def ask_model(provider, client=None, model=None, timeout=TIMEOUT, vision=True):
    """One chat call. Tries vision (reference images attached) + json_schema, degrading to
    text-only / json_object / plain on API errors. Returns (raw_dict, info)."""
    from . import agent_planner as ap
    model = model or ap.default_model(provider)
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=ap.PROVIDERS[provider][0], api_key=ap.key_for(provider),
                        timeout=timeout, max_retries=0)
    imgs = _images() if vision else []
    fmts = [{"type": "json_schema", "json_schema": {"name": "legionary_theme", "schema": THEME_SCHEMA}},
            {"type": "json_object"}, None]
    attempts = ([(True, f) for f in fmts[:1]] if imgs else []) + [(False, f) for f in fmts]
    t0, last = time.time(), None
    for use_img, fmt in attempts:
        if time.time() - t0 > timeout:
            break
        content = PROMPT if not use_img else (
            [{"type": "text", "text": PROMPT}] + [{"type": "image_url", "image_url": {"url": u}} for u in imgs])
        kw = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 6000,
              "timeout": max(1.0, timeout - (time.time() - t0))}
        if fmt:
            kw["response_format"] = fmt
        try:
            r = client.chat.completions.create(**kw)
            raw = extract_json(r.choices[0].message.content)
        except Exception as e:  # noqa: BLE001 - degrade and retry
            last = f"{type(e).__name__}: {str(e)[:160]}"
            continue
        u = getattr(r, "usage", None)
        pin, pout = ap.PRICES.get(provider, (ap.PRICE_IN, ap.PRICE_OUT))
        tin, tout = (getattr(u, "prompt_tokens", 0) or 0), (getattr(u, "completion_tokens", 0) or 0)
        return raw, {"provider": provider, "model": model, "vision": use_img,
                     "response_format": fmt["type"] if fmt else "none", "seconds": round(time.time() - t0, 1),
                     "tokens_in": tin, "tokens_out": tout, "cost_usd": round((tin * pin + tout * pout) / 1e6, 5)}
    raise RuntimeError(last or "timeout")


class Director:
    """Holds the active theme. start() returns immediately (cache or defaults) and swaps in the
    model theme from a background thread when it arrives; poll() reports a swap to the viewer."""

    def __init__(self, backend="auto", refresh=False, cache_path=CACHE_PATH, timeout=TIMEOUT, clients=None):
        self.backend, self.refresh, self.cache_path, self.timeout = backend, refresh, cache_path, timeout
        self.clients = clients or {}
        self.theme, self.label, self.info, self.errors = copy.deepcopy(DEFAULTS), "theme: default", {}, []
        self._new, self._lock, self.thread = None, threading.Lock(), None

    def start(self, background=True):
        if self.backend == "off":
            self.label = "theme: default (director off)"
            return self
        cached, meta = load_cache(self.cache_path)
        if cached:
            self.theme = cached
            self.label = f"theme: {_short(meta.get('model'))} via {meta.get('provider')} (cached)"
            self.info = {k: v for k, v in meta.items() if k != "theme"}
            if not self.refresh:
                return self
        if background:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        else:
            self._run()
        return self

    def _run(self):
        from . import agent_planner as ap
        try:
            order = list(self.clients) if self.clients else ap.select_backends(self.backend)
        except Exception as e:  # noqa: BLE001
            self.errors.append(str(e)); return
        for prov in order:
            try:
                raw, info = ask_model(prov, self.clients.get(prov), timeout=self.timeout)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"{prov}: {e}"); continue
            theme, warn = validate(raw)
            info["warnings"] = warn
            try:
                save_cache(theme, info, self.cache_path)
            except OSError as e:
                self.errors.append(f"cache: {e}")
            with self._lock:
                self._new = (theme, f"theme: {_short(info['model'])} via {prov}", info)
            return

    def poll(self):
        """Apply a freshly arrived model theme; True if the theme changed this call."""
        with self._lock:
            new, self._new = self._new, None
        if not new:
            return False
        self.theme, self.label, self.info = new
        return True


def _short(model):
    return str(model or "?").rsplit("/", 1)[-1]
