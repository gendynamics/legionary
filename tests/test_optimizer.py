import json
from types import SimpleNamespace as NS
import pytest
from quantum_optimizer import tools, planner
from quantum_optimizer.benchmarks import all_benchmarks
from quantum_optimizer.agents import swarm
from quantum_optimizer.reviewer import review
from quantum_optimizer.schemas import TOOLS

BELL = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'
BELL_PAD = BELL + "x q[1];\nx q[1];\n"
BELL_BAD = BELL + "x q[1];\n"


def test_schemas():
    assert [t["name"] for t in TOOLS] == ["count_gates", "check_equiv", "simulate"]
    assert all(t["parameters"]["additionalProperties"] is False for t in TOOLS)


def test_count_gates():
    m = tools.count_gates(BELL_PAD)
    assert m["total"] == 4 and m["cx"] == 1 and m["depth"] == 4


def test_equiv():
    assert tools.check_equiv(BELL, BELL)["ok"]
    assert tools.check_equiv(BELL_PAD, BELL)["ok"]
    assert not tools.check_equiv(BELL, BELL_BAD)["ok"]


def test_simulate():
    c = tools.simulate(BELL, shots=500, seed=1)["counts"]
    assert sum(c.values()) == 500
    assert set(c) <= {"00", "11"}


def test_benchmarks_build():
    b = all_benchmarks()
    assert set(b) == {"qft4", "grover3", "bv4"}
    for q in b.values():
        assert tools.count_gates(q)["total"] > 0


@pytest.mark.parametrize("name", ["qft4", "grover3", "bv4"])
def test_swarm_reviewer(name):
    q = all_benchmarks()[name]
    best = review(q, swarm(q))
    assert tools.check_equiv(q, best["qasm"])["ok"]
    assert tools.count_gates(best["qasm"])["total"] <= tools.count_gates(q)["total"]


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return NS(id="r1", output_text="", output=[NS(type="function_call", call_id="c1",
                      name="count_gates", arguments=json.dumps({"qasm": BELL}))])
        return NS(id="r2", output_text="done", output=[NS(type="message")])


def test_planner_loop():
    fake = FakeResponses()
    out = planner.plan("go", ai=NS(responses=fake))
    assert out == "done"
    assert "tool_choice" not in fake.calls[0]
    assert fake.calls[0]["model"] == "muse-spark-1.3"
    second = fake.calls[1]
    assert second["previous_response_id"] == "r1"
    item = second["input"][0]
    assert item["type"] == "function_call_output" and item["call_id"] == "c1"
    assert json.loads(item["output"])["cx"] == 1


def test_run_error():
    assert "error" in planner.run("count_gates", {"qasm": "bad"})
