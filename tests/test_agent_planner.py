import pytest
"""Offline mocked tests for the Muse Spark agent tool loop (no network)."""
import json
from types import SimpleNamespace as NS
from quantum_optimizer import agent_planner as ap
from quantum_optimizer.benchmarks import all_benchmarks

G = all_benchmarks()["grover3"]
BAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nx q[0];\n'


def fc(i, name, **args):
    return NS(type="function_call", call_id=f"k{i}", name=name, arguments=json.dumps(args))


class FakeResponses:
    """Scripted model: each step is a list of calls, or a callable(prev_outputs) -> calls."""
    def __init__(self, script):
        self.script, self.i, self.inputs = script, 0, []
    def create(self, **kw):
        self.inputs.append(kw.get("input"))
        step = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        calls = step(kw.get("input")) if callable(step) else step
        return NS(id=f"r{self.i}", output=calls, output_text="done",
                  usage=NS(input_tokens=100, output_tokens=10))


def last(outs):
    return json.loads(outs[-1]["output"])


def test_dispatch_chain_and_finish():
    ai = NS(responses=FakeResponses([
        [fc(1, "list_strategies")],
        [fc(2, "run_pass", circuit_id="c0", strategy="level1")],
        lambda o: [fc(3, "run_pass", circuit_id=last(o)["circuit_id"], strategy="level2")],
        lambda o: [fc(4, "finish", circuit_id=last(o)["circuit_id"])],
    ]))
    r = ap.run_agent("grover3", G, ai=ai)
    assert r["api"] == "responses" and r["source"] == "model" and r["equiv"]
    assert r["chain"] == ["level1", "level2"] and r["gates"] == 81 and r["cx"] == 24
    assert [c["name"] for c in r["tool_calls"]] == ["list_strategies", "run_pass", "run_pass", "finish"]
    assert r["usage"] == {"input": 400, "output": 40} and r["cost_usd"] > 0


def test_rejects_non_equivalent_finish():
    s = ap.Session(G)
    s.store["cbad"] = {"qasm": BAD, "chain": ["evil"]}
    out = s.dispatch("finish", {"circuit_id": "cbad"})
    assert out["accepted"] is False and s.final is None
    s.final = "cbad"  # even if forced, result() re-checks in code
    r = s.result()
    assert r["equiv"] and r["source"] == "fallback-best-equivalent" and r["circuit_id"] == "c0"


def test_respects_tool_call_cap():
    ai = NS(responses=FakeResponses([[fc(1, "metrics", circuit_id="c0")]]))  # loops forever
    r = ap.run_agent("grover3", G, ai=ai, max_calls=5, turns=50)
    assert r["n_tool_calls"] == 5 and ai.responses.i <= 6
    assert ap.Session(G, max_calls=0).dispatch("metrics", {"circuit_id": "c0"})["error"].startswith("tool-call cap")


def test_bad_args_and_chat_fallback():
    s = ap.Session(G)
    assert "error" in s.dispatch("run_pass", {"circuit_id": "nope", "strategy": "level1"})
    assert "error" in s.dispatch("run_pass", {"circuit_id": "c0", "strategy": "rm -rf"})

    class Boom:
        def create(self, **kw): raise RuntimeError("tools not supported")
    steps = iter([
        [NS(id="t1", type="function", function=NS(name="run_pass", arguments=json.dumps({"circuit_id": "c0", "strategy": "level2"})))],
        [NS(id="t2", type="function", function=NS(name="finish", arguments=json.dumps({"circuit_id": "c1"})))],
    ])
    chat = NS(create=lambda **kw: NS(usage=NS(prompt_tokens=5, completion_tokens=1),
                                     choices=[NS(message=NS(content="", tool_calls=next(steps)))]))
    r = ap.run_agent("grover3", G, ai=NS(responses=Boom(), chat=NS(completions=chat)))
    assert r["api"] == "chat" and r["errors"] and r["chain"] == ["level2"] and r["equiv"]


def _clear(mp):
    for e in ("MODEL_API_KEY", "META_API_KEY", "NEBIUS_API_KEY", "FIREWORKS_API_KEY", "FIREWORKS_MODEL"):
        mp.delenv(e, raising=False)


def test_backend_selection(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("NEBIUS_API_KEY", "x")
    assert ap.select_backends("auto") == ["nebius"]
    monkeypatch.setenv("META_API_KEY", "y")
    assert ap.select_backends("auto") == ["meta", "nebius"]
    assert ap.select_backends("nebius") == ["nebius"]
    monkeypatch.delenv("META_API_KEY"); monkeypatch.setenv("MODEL_API_KEY", "z")
    assert ap.pick_provider() == "meta"
    _clear(monkeypatch)
    import pytest
    with pytest.raises(RuntimeError):
        ap.select_backends("meta")


def test_auto_runtime_fallback_to_nebius(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MODEL_API_KEY", "y"); monkeypatch.setenv("NEBIUS_API_KEY", "x")
    class Boom:
        def create(self, **kw): raise RuntimeError("401 bad key")
    neb = NS(responses=FakeResponses([[fc(1, "run_pass", circuit_id="c0", strategy="level2")],
                                      [fc(2, "finish", circuit_id="c1")]]))
    r = ap.run_agent("grover3", G, clients={"meta": NS(responses=Boom()), "nebius": neb})
    assert r["provider"] == "nebius" and r["backends_tried"] == ["meta", "nebius"]
    assert r["source"] == "model" and any("meta" in e or "401" in e for e in r["errors"])


def test_finish_nudge():
    ai = NS(responses=FakeResponses([[fc(1, "metrics", circuit_id="c0")]] * 4 +
                                    [[fc(9, "finish", circuit_id="c0")]]))
    r = ap.run_agent("grover3", G, ai=ai, max_calls=6)
    nudges = [m for inp in ai.responses.inputs for m in inp
              if isinstance(m, dict) and m.get("role") == "user" and "remain" in m.get("content", "")]
    assert r["nudged"] and nudges and r["source"] == "model"
    import inspect; assert inspect.signature(ap.run_agent).parameters["max_calls"].default == 50


# ---- LangSmith tracing (offline: env monkeypatched, client mocked, sockets blocked) ----
def _script():
    return FakeResponses([
        [fc(1, "run_pass", circuit_id="c0", strategy="level1")],
        lambda o: [fc(2, "run_pass", circuit_id=last(o)["circuit_id"], strategy="level2")],
        lambda o: [fc(3, "finish", circuit_id=last(o)["circuit_id"])],
    ])


def _no_network(monkeypatch):
    import socket
    def boom(*a, **k):
        raise AssertionError("network call attempted in test")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


def test_tracing_disabled_without_env(monkeypatch):
    pytest.importorskip("langsmith")
    _no_network(monkeypatch)
    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert not ap.tracing_enabled()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")  # key still missing -> still off
    assert not ap.tracing_enabled()
    called = []
    monkeypatch.setattr("langsmith.traceable", lambda *a, **k: called.append(1))
    r = ap.run_agent("grover3", G, ai=NS(responses=_script()))
    assert r["gates"] == 81 and r["source"] == "model" and not called


def test_tracing_enabled_with_env(monkeypatch):
    from unittest.mock import MagicMock
    langsmith = pytest.importorskip("langsmith")
    _no_network(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-not-a-real-key")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert ap.tracing_enabled() and ap.tracing_project() == "legionary"
    client = MagicMock()
    with langsmith.tracing_context(enabled=True, client=client):
        r = ap.run_agent("grover3", G, ai=NS(responses=_script()), provider="nebius")
    assert r["gates"] == 81 and r["equiv"] and r["source"] == "model"
    runs = [c.kwargs for c in client.create_run.call_args_list]
    names = [x.get("name") for x in runs]
    assert "legionary.agent_planner" in names
    assert names.count("run_pass") == 2 and "finish" in names
    top = next(x for x in runs if x["name"] == "legionary.agent_planner")
    md = top.get("extra", {}).get("metadata", {})
    assert md.get("backend") == "nebius" and md.get("circuit") == "grover3" and md.get("max_calls") == 50
    assert top.get("session_name") == "legionary"
    ends = [c.kwargs for c in client.update_run.call_args_list]
    outs = [e.get("outputs") or {} for e in ends]
    assert any(o.get("gates") == 81 and o.get("chain") == ["level1", "level2"] and "nudged" in o for o in outs)


# ---- Fireworks backend (offline, mocked chat client) ----
def _tc(i, name, **args):
    return NS(id=f"t{i}", type="function", function=NS(name=name, arguments=json.dumps(args)))


class FakeChat:
    def __init__(self, script):
        self.script, self.i, self.kw = script, 0, []
    def create(self, **kw):
        self.kw.append(kw)
        step = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        calls = step(kw["messages"]) if callable(step) else step
        return NS(usage=NS(prompt_tokens=200, completion_tokens=50),
                  choices=[NS(message=NS(content="", tool_calls=calls or None))])


def test_fireworks_key_and_model(monkeypatch):
    import pytest
    _clear(monkeypatch)
    with pytest.raises(RuntimeError):
        ap.select_backends("fireworks")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")
    assert ap.select_backends("fireworks") == ["fireworks"] == ap.select_backends("auto")
    assert ap.key_for("fireworks") == "fw-test"
    assert ap.PROVIDERS["fireworks"][0] == "https://api.fireworks.ai/inference/v1"
    assert ap.default_model("fireworks") == "accounts/fireworks/models/muse-spark-open"
    monkeypatch.setenv("FIREWORKS_MODEL", "accounts/x/models/y")
    assert ap.default_model("fireworks") == "accounts/x/models/y"


def test_auto_order_meta_fireworks_nebius(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("NEBIUS_API_KEY", "n"); monkeypatch.setenv("FIREWORKS_API_KEY", "f")
    assert ap.select_backends("auto") == ["fireworks", "nebius"]
    monkeypatch.setenv("META_API_KEY", "m")
    assert ap.select_backends("auto") == ["meta", "fireworks", "nebius"]


def test_fireworks_chat_tool_loop_and_fallback(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("META_API_KEY", "m"); monkeypatch.setenv("FIREWORKS_API_KEY", "f")
    class Boom:
        def create(self, **kw): raise RuntimeError("401 bad key")
    tool_msg = lambda msgs: json.loads([m for m in msgs if m["role"] == "tool"][-1]["content"])
    chat = FakeChat([
        [_tc(1, "run_pass", circuit_id="c0", strategy="level1")],
        lambda m: [_tc(2, "run_pass", circuit_id=tool_msg(m)["circuit_id"], strategy="level2")],
        lambda m: [_tc(3, "check_equivalence", circuit_id=tool_msg(m)["circuit_id"])],
        lambda m: [_tc(4, "finish", circuit_id="c2")],
    ])
    fw = NS(responses=Boom(), chat=NS(completions=chat))  # responses must NOT be used
    r = ap.run_agent("grover3", G, clients={"meta": NS(responses=Boom()), "fireworks": fw})
    assert r["provider"] == "fireworks" and r["backends_tried"] == ["meta", "fireworks"]
    assert r["api"] == "chat" and r["source"] == "model" and r["equiv"] and r["gates"] == 81
    assert r["model"] == "accounts/fireworks/models/muse-spark-open"
    kw = chat.kw[-1]
    assert kw["max_tokens"] >= 4096 and len(kw["tools"]) == 5
    tools = [m for m in kw["messages"] if m["role"] == "tool"]
    assert [t["tool_call_id"] for t in tools] == ["t1", "t2", "t3", "t4"]
    assert r["usage"] == {"input": 800, "output": 200}
    assert abs(r["cost_usd"] - (800 * 1.25 + 200 * 4.25) / 1e6) < 1e-12


def test_fireworks_chat_cap_nudge_fallback(monkeypatch):
    _clear(monkeypatch)
    chat = FakeChat([[_tc(1, "metrics", circuit_id="c0")]])  # never finishes
    r = ap.run_agent("grover3", G, clients={"fireworks": NS(chat=NS(completions=chat))},
                     provider="fireworks", max_calls=5, turns=50)
    assert r["n_tool_calls"] == 5 and r["nudged"] and r["source"] == "fallback-best-equivalent"
    assert any(m["role"] == "user" and "tool calls remain" in m["content"] for m in chat.kw[-1]["messages"])


def test_nebius_and_meta_model_override(monkeypatch):
    from quantum_optimizer import agent_planner as ap
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    assert ap.default_model("nebius") == "meta-models/muse-spark-open"
    monkeypatch.setenv("NEBIUS_MODEL", "meta-models/muse-spark-open-1026")
    assert ap.default_model("nebius") == "meta-models/muse-spark-open-1026"
    monkeypatch.setenv("META_MODEL", "muse-spark-x")
    assert ap.default_model("meta") == "muse-spark-x"
