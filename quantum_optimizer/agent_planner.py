"""Agentic planner: Muse Spark drives the optimization with real function tools.

The model picks the strategy sequence itself via run_pass / metrics / check_equivalence /
list_strategies / finish. Circuits live in a server-side store keyed by id. The final
circuit is re-verified with tools.check_equiv against the ORIGINAL in code; a
non-equivalent finish is rejected, and if the model never finishes validly we fall back
to the best equivalent circuit seen (or the original).
Backend (--backend auto|meta|fireworks|nebius): meta = api.meta.ai muse-spark-1.3 (key MODEL_API_KEY or
META_API_KEY); fireworks = accounts/fireworks/models/muse-spark-open (FIREWORKS_API_KEY, model
overridable via FIREWORKS_MODEL) over Chat Completions; nebius = Token Factory
meta-models/muse-spark-open (NEBIUS_API_KEY, model overridable via NEBIUS_MODEL; META_MODEL for meta). auto order: meta -> fireworks -> nebius (keyed ones
only), falling back at runtime on error. Others try Responses first, then chat.completions.
"""
import json, os, logging
from .agents import STRATEGIES, work
from .tools import count_gates, check_equiv

PROVIDERS = {
    "meta":   ("https://api.meta.ai/v1", ("MODEL_API_KEY", "META_API_KEY"), "muse-spark-1.3"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", ("FIREWORKS_API_KEY",),
                  "accounts/fireworks/models/muse-spark-open"),
    "nebius": ("https://api.tokenfactory.nebius.com/v1", ("NEBIUS_API_KEY",), "meta-models/muse-spark-open"),
}
# Tool-call cap: raised 30 -> 50 (2026-10-31). At 30 the Fireworks/Nebius Grover runs spent every
# call exploring equal-scoring variants and the result came from the fallback, not the model's own
# finish. 50 leaves headroom for check_equivalence + finish; turns stays above the cap.
DEFAULT_MAX_TOOL_CALLS = 50
DEFAULT_TURNS = 60
NUDGE_AT = 5  # remaining tool calls at which we tell the model to finish (was 3)
PRICE_IN, PRICE_OUT = 1.25, 4.25  # $ per 1M tokens (default)
# Per-backend $/1M (input, output). Fireworks serverless muse-spark-open: $1.25 in / $4.25 out
# (fireworks.ai/models/fireworks/muse-spark-open, checked 2026-10-31).
PRICES = {"meta": (1.25, 4.25), "nebius": (1.25, 4.25), "fireworks": (1.25, 4.25)}
BACKENDS = ("meta", "fireworks", "nebius")  # auto order
# Fireworks Responses tool calling was seen emitting raw template text instead of function_call
# items, which ends the loop silently (no exception to fall back on), so use Chat Completions.
FORCE_API = {"meta": "responses", "fireworks": "chat"}
MAX_TOKENS = {"fireworks": 8192}  # reasoning model: leave room for reasoning + tool call

_P = lambda props, req: {"type": "object", "properties": props, "required": req,
                         "additionalProperties": False}
_CID = {"circuit_id": {"type": "string"}}
FUNCS = [
    ("list_strategies", "List available optimization strategies.", _P({}, [])),
    ("run_pass", "Apply one strategy to a stored circuit; returns new circuit_id and metrics.",
     _P({**_CID, "strategy": {"type": "string", "enum": list(STRATEGIES)}}, ["circuit_id", "strategy"])),
    ("metrics", "Gate count, depth and CX count of a stored circuit.", _P(_CID, ["circuit_id"])),
    ("check_equivalence", "Check a stored circuit is unitarily equivalent to the ORIGINAL.",
     _P(_CID, ["circuit_id"])),
    ("finish", "Submit the final circuit_id. Rejected unless equivalent to the original.",
     _P(_CID, ["circuit_id"])),
]
RESP_TOOLS = [{"type": "function", "name": n, "description": d, "parameters": p} for n, d, p in FUNCS]
CHAT_TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": p}}
              for n, d, p in FUNCS]


MODEL_ENV = {"meta": "META_MODEL", "fireworks": "FIREWORKS_MODEL", "nebius": "NEBIUS_MODEL"}


def default_model(provider):
    """Per-backend model override, e.g. NEBIUS_MODEL=meta-models/muse-spark-open-1026."""
    env = MODEL_ENV.get(provider)
    if env and os.environ.get(env):
        return os.environ[env]
    return PROVIDERS[provider][2]


def key_for(provider):
    return next((os.environ[e] for e in PROVIDERS[provider][1] if os.environ.get(e)), None)


def select_backends(backend="auto"):
    """Ordered list of backends to try. auto: meta -> fireworks -> nebius (keyed only)."""
    if backend in BACKENDS:
        if not key_for(backend):
            raise RuntimeError(f"--backend {backend}: none of {PROVIDERS[backend][1]} is set")
        return [backend]
    order = [b for b in BACKENDS if key_for(b)]
    if not order:
        raise RuntimeError("no Meta (MODEL_API_KEY/META_API_KEY), FIREWORKS_API_KEY or NEBIUS_API_KEY key set")
    return order


def pick_provider():
    return select_backends("auto")[0]


from . import events


class Session:
    """Server-side circuit store + tool dispatch."""
    def __init__(self, original, max_calls=20):
        self.original, self.max_calls = original, max_calls
        self.store = {"c0": {"qasm": original, "chain": []}}
        self.calls, self.final, self.n, self.nudged = [], None, 0, False

    def nudge(self):
        """One-time message once <= NUDGE_AT calls remain and no finish yet."""
        if self.final or self.nudged or self.max_calls - len(self.calls) > NUDGE_AT:
            return None
        self.nudged = True
        left = max(self.max_calls - len(self.calls), 0)
        return (f"Only {left} tool calls remain. Stop exploring: call check_equivalence on your best "
                f"circuit_id and then call finish with it now.")

    def _m(self, cid):
        m = count_gates(self.store[cid]["qasm"])
        return {"gates": m["total"], "depth": m["depth"], "cx": m["cx"]}

    def dispatch(self, name, args):
        if len(self.calls) >= self.max_calls:
            out = {"error": f"tool-call cap ({self.max_calls}) reached; stop"}
            self.calls.append({"name": name, "args": args, "result": out, "capped": True})
            return out
        a = args or {}
        events.emit("tool_call", call=len(self.calls) + 1, max_calls=self.max_calls, tool=name,
                    args=a, strategy=a.get("strategy"), circuit_id=a.get("circuit_id"))
        try:
            out = self._run(name, a)
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        self.calls.append({"name": name, "args": args, "result": out})
        if events.enabled() and isinstance(out, dict):
            if name == "run_pass" and "circuit_id" in out:
                events.emit_circuit("circuit_update", self.store[out["circuit_id"]]["qasm"],
                                    circuit_id=out["circuit_id"], parent_id=a.get("circuit_id"),
                                    chain=out.get("chain"))
            elif name == "check_equivalence" and "equivalent" in out:
                events.emit("equivalence", circuit_id=a.get("circuit_id"), ok=out["equivalent"],
                            status="pass" if out["equivalent"] else "fail")
            elif name == "finish" and "accepted" in out:
                events.emit("equivalence", circuit_id=a.get("circuit_id"), ok=out["accepted"], finish=True,
                            status="pass" if out["accepted"] else "fail")
        return out

    def _run(self, name, a):
        cid = a.get("circuit_id")
        if name != "list_strategies" and cid not in self.store:
            if name in ("run_pass", "metrics", "check_equivalence", "finish"):
                return {"error": f"unknown circuit_id {cid!r}; known: {sorted(self.store)}"}
        if name == "list_strategies":
            return {"strategies": list(STRATEGIES), "start_circuit_id": "c0"}
        if name == "run_pass":
            s = a.get("strategy")
            if s not in STRATEGIES:
                return {"error": f"unknown strategy {s!r}"}
            r = work(s, self.store[cid]["qasm"])
            if "error" in r:
                return {"error": r["error"]}
            self.n += 1
            new = f"c{self.n}"
            self.store[new] = {"qasm": r["qasm"], "chain": self.store[cid]["chain"] + [s]}
            return {"circuit_id": new, "chain": " -> ".join(self.store[new]["chain"]), **self._m(new)}
        if name == "metrics":
            return self._m(cid)
        if name == "check_equivalence":
            return {"equivalent": self._eq(cid)}
        if name == "finish":
            if not self._eq(cid):
                return {"accepted": False, "error": "not equivalent to the original; pick another circuit"}
            self.final = cid
            return {"accepted": True, **self._m(cid)}
        return {"error": f"unknown tool {name!r}"}

    def _eq(self, cid):
        return check_equiv(self.original, self.store[cid]["qasm"])["ok"]

    def result(self):
        """Enforced in code: final must be equivalent; else best equivalent seen; else original."""
        cid, src = self.final, "model"
        if cid is None or not self._eq(cid):
            ok = [c for c in self.store if self._eq(c)]
            cid = min(ok, key=lambda c: (self._m(c)["gates"], self._m(c)["cx"], self._m(c)["depth"]))
            src = "fallback-best-equivalent"
        return {"circuit_id": cid, "chain": self.store[cid]["chain"], "qasm": self.store[cid]["qasm"],
                "equiv": self._eq(cid), "source": src, **self._m(cid)}


SYSTEM = ("You are the Legionary planner for quantum circuit optimization. The original circuit "
          "is stored as circuit_id 'c0'. Use the tools to minimize gate count (then CX, then depth). "
          "You choose which strategies to apply and in what order; passes can be chained (run_pass on "
          "a circuit_id returned by an earlier run_pass). Work in rounds: apply passes to your current best "
          "circuit; as soon as a round of passes brings no improvement in gate count, call "
          "check_equivalence on your best candidate and then call finish with that circuit_id. You MUST "
          "end by calling finish. You have at most {cap} tool calls; keep 2 in reserve for "
          "check_equivalence + finish.")


def _usage(u, tot):
    if u is None:
        return
    g = lambda *ks: next((getattr(u, k) for k in ks if getattr(u, k, None) is not None), 0)
    tot["input"] += g("input_tokens", "prompt_tokens")
    tot["output"] += g("output_tokens", "completion_tokens")


def _loop_responses(ai, model, sess, prompt, turns, tot):
    # Stateless: resend the full item history each turn (Nebius rejects previous_response_id).
    hist = [{"role": "user", "content": prompt[1]}]
    r = ai.responses.create(model=model, instructions=prompt[0], input=hist, tools=RESP_TOOLS)
    _usage(r.usage, tot)
    for _ in range(turns):
        calls = [x for x in r.output if x.type == "function_call"]
        if not calls or sess.final or len(sess.calls) >= sess.max_calls:
            return getattr(r, "output_text", "")
        for c in calls:
            hist.append({"type": "function_call", "call_id": c.call_id, "name": c.name,
                         "arguments": c.arguments or "{}"})
            hist.append({"type": "function_call_output", "call_id": c.call_id,
                         "output": json.dumps(sess.dispatch(c.name, json.loads(c.arguments or "{}")))})
        if sess.final or len(sess.calls) >= sess.max_calls:
            return ""
        n = sess.nudge()
        if n:
            hist.append({"role": "user", "content": n})
        r = ai.responses.create(model=model, instructions=prompt[0], input=list(hist), tools=RESP_TOOLS)
        _usage(r.usage, tot)
    return getattr(r, "output_text", "")


def _loop_chat(ai, model, sess, prompt, turns, tot, max_tokens=None):
    msgs = [{"role": "system", "content": prompt[0]}, {"role": "user", "content": prompt[1]}]
    extra = {"max_tokens": max_tokens} if max_tokens else {}
    for _ in range(turns):
        r = ai.chat.completions.create(model=model, messages=msgs, tools=CHAT_TOOLS, **extra)
        _usage(r.usage, tot)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content or ""
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [{"id": t.id, "type": "function",
                                     "function": {"name": t.function.name, "arguments": t.function.arguments}}
                                    for t in m.tool_calls]})
        for t in m.tool_calls:
            out = sess.dispatch(t.function.name, json.loads(t.function.arguments or "{}"))
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps(out)})
        if sess.final or len(sess.calls) >= sess.max_calls:
            return ""
        n = sess.nudge()
        if n:
            msgs.append({"role": "user", "content": n})
    return ""


# ---- LangSmith tracing (opt-in; no-op unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY set) ----
_log = logging.getLogger(__name__)


def tracing_enabled():
    return (os.environ.get("LANGSMITH_TRACING", "").strip().lower() == "true"
            and bool(os.environ.get("LANGSMITH_API_KEY")))


def tracing_project():
    return os.environ.get("LANGSMITH_PROJECT") or "legionary"


def _traceable(fn, **kw):
    """Wrap fn with langsmith.traceable when tracing is on; never raises."""
    if not tracing_enabled():
        return fn
    try:
        from langsmith import traceable
        return traceable(project_name=tracing_project(), **kw)(fn)
    except Exception as e:  # tracing must never break a run
        _log.warning("LangSmith tracing disabled: %s", e)
        return fn


def _trace_client(ai):
    """wrap_openai so each model call shows up as an llm run; falls back to the raw client."""
    if not tracing_enabled():
        return ai
    try:
        from openai import OpenAI
        if not isinstance(ai, OpenAI):
            return ai
        from langsmith.wrappers import wrap_openai
        return wrap_openai(ai, tracing_extra={"project_name": tracing_project()})
    except Exception as e:
        _log.warning("LangSmith wrap_openai skipped: %s", e)
        return ai


def _client(provider):
    from openai import OpenAI
    return OpenAI(base_url=PROVIDERS[provider][0], api_key=key_for(provider))


def run_agent(name, original, ai=None, model=None, provider=None, max_calls=DEFAULT_MAX_TOOL_CALLS, turns=DEFAULT_TURNS, api="auto",
              backend=None, clients=None):
    """Returns dict with chain/metrics/equiv plus tool_calls, usage, cost, api, provider, errors.

    ai/provider: run a single given client (tests). Otherwise backend auto|meta|fireworks|nebius picks
    the order; clients={provider: client} overrides construction (tests)."""
    if ai is not None:
        return _run_one(name, original, ai, model or "mock", provider, max_calls, turns, api)
    order = [provider] if provider else select_backends(backend or "auto")
    errors, r = [], None
    for prov in order:
        cl = (clients or {}).get(prov) or _client(prov)
        r = _run_one(name, original, cl, model or default_model(prov), prov, max_calls, turns,
                     FORCE_API.get(prov, api))
        r["errors"] = errors + r["errors"]
        if r["api"] and "errored" not in r["api"]:
            break
        errors = r["errors"] + [f"backend {prov} failed; falling back" if prov != order[-1] else ""]
        errors = [e for e in errors if e]
    r["backends_tried"] = order[:order.index(r["provider"]) + 1]
    return r


def _run_one(name, original, ai, model, provider, max_calls, turns, api):
    if not tracing_enabled():
        return _run_one_inner(name, original, ai, model, provider, max_calls, turns, api)
    meta = {"backend": provider, "model": model, "circuit": name, "max_calls": max_calls, "api": api}
    fn = _traceable(_run_one_inner, name="legionary.agent_planner", run_type="chain",
                    metadata=meta, tags=["legionary", f"backend:{provider}"],
                    process_inputs=lambda i: {k: i.get(k) for k in ("name", "model", "provider", "max_calls", "api")},
                    process_outputs=_trace_outputs)
    return fn(name, original, _trace_client(ai), model, provider, max_calls, turns, api)


def _trace_outputs(r):
    try:
        return {k: r.get(k) for k in ("strategy", "chain", "gates", "depth", "cx", "equiv", "source",
                                      "nudged", "n_tool_calls", "usage", "cost_usd", "provider",
                                      "model", "api", "errors")}
    except Exception:
        return {"output": str(r)[:2000]}


def _run_one_inner(name, original, ai, model, provider, max_calls, turns, api):
    sess, tot, errors = Session(original, max_calls), {"input": 0, "output": 0}, []
    if tracing_enabled():  # one tool-type run per dispatch, named after the tool
        raw = sess.dispatch
        def _d(tool, args):
            return _traceable(lambda tool_name, args: raw(tool_name, args), name=tool, run_type="tool")(tool, args)
        sess.dispatch = _d
    m0 = count_gates(original)
    prompt = (SYSTEM.format(cap=max_calls),
              f"Circuit '{name}': {m0['total']} gates, depth {m0['depth']}, {m0['cx']} CX. "
              f"Strategies: {', '.join(STRATEGIES)}. Optimize it.")
    used, text = None, ""
    order = {"auto": ["responses", "chat"], "responses": ["responses"], "chat": ["chat"]}[api]
    for kind in order:
        try:
            if kind == "responses":
                text = _loop_responses(ai, model, sess, prompt, turns, tot)
            else:
                text = _loop_chat(ai, model, sess, prompt, turns, tot, MAX_TOKENS.get(provider))
            used = kind
            break
        except Exception as e:
            errors.append(f"{kind}: {type(e).__name__}: {getattr(e, 'status_code', '')} {e}"[:600])
            if sess.calls:  # partial progress: don't restart with a different API
                used = kind + " (errored)"
                break
    res = sess.result()
    pin, pout = PRICES.get(provider, (PRICE_IN, PRICE_OUT))
    cost = tot["input"] / 1e6 * pin + tot["output"] / 1e6 * pout
    return {**res, "strategy": " -> ".join(res["chain"]) or "original", "tool_calls": sess.calls,
            "n_tool_calls": len(sess.calls), "usage": tot, "cost_usd": cost, "api": used,
            "provider": provider, "model": model, "errors": errors, "final_text": text,
            "nudged": sess.nudged}
