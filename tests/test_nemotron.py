import json
from types import SimpleNamespace as NS
import pytest
from quantum_optimizer import nemotron
from quantum_optimizer import reviewer as rv
from quantum_optimizer.agents import swarm
from quantum_optimizer.benchmarks import all_benchmarks


class Fake:
    def __init__(self, content=None, exc=None):
        self.calls, self.content, self.exc = [], content, exc
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return NS(choices=[NS(message=NS(content=self.content))])


Q = next(iter(all_benchmarks().values()))


def test_approve():
    f = Fake('```json\n' + json.dumps({"verdict": "approve", "reasons": ["ok"], "risk_notes": []}) + '\n```')
    r = nemotron.review_rewrite(Q, Q, "x", ai=f)
    assert r["status"] == "ok" and r["verdict"] == "approve" and r["reasons"] == ["ok"]
    assert f.calls[0]["model"] == "nvidia/nemotron-3-super-120b-a12b"


def test_reject():
    r = nemotron.review_rewrite(Q, Q, ai=Fake('{"verdict":"REJECT","reasons":"bad","risk_notes":["phase"]}'))
    assert r["verdict"] == "reject" and r["reasons"] == ["bad"] and r["risk_notes"] == ["phase"]


def test_malformed():
    r = nemotron.review_rewrite(Q, Q, ai=Fake("not json at all"))
    assert r["status"] == "error" and r["verdict"] is None


def test_missing_key(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert nemotron.review_rewrite(Q, Q)["status"] == "skipped"


def test_api_error():
    r = nemotron.review_rewrite(Q, Q, ai=Fake(exc=RuntimeError("boom")))
    assert r["status"] == "error" and r["error"] == "RuntimeError"


def test_toggle(monkeypatch):
    monkeypatch.setenv("LEGIONARY_NEMOTRON_REVIEW", "0")
    assert not nemotron.enabled() and nemotron.enabled(True)


def test_order_and_record(monkeypatch):
    order, real = [], rv.check_equiv
    monkeypatch.setattr(rv, "check_equiv", lambda a, b: (order.append("equiv"), real(a, b))[1])
    def second(b, a, s):
        order.append("nemotron")
        return {"status": "ok", "verdict": "reject"}
    recs, cands = [], swarm(Q)
    best_with = rv.review(Q, cands, second_opinion=second, records=recs)
    assert order[:2] == ["nemotron", "equiv"] and order.count("nemotron") == order.count("equiv")
    assert all("nemotron" in r and "equiv" in r for r in recs) and recs
    # advisory: a reject verdict does not change the equivalence-based choice
    assert best_with["strategy"] == rv.review(Q, cands)["strategy"]
