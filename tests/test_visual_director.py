"""Visual director (Muse Spark theme) tests. The model client is always mocked; no network."""
import json, types

import pytest

from quantum_optimizer import visual_director as vd, viewer


class FakeClient:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        msg = types.SimpleNamespace(content=self.reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=50))


GOOD = {"palette": {"background": "#010203", "window_frame": "#40E0D0", "cx_link": "#ff00ff"},
        "flame": {"core": "#ffffff", "mid": "#3366ff", "outer": "#112244", "intensity": 9, "height": -1},
        "still_cube": {"color": "#808080", "alpha": 0.4, "edge_style": "dashed", "fill": "#000000"},
        "cx_ribbon": {"enabled": True, "colors": ["#00ffff", "#ff00ff"], "base_width": 3, "max_width": 1},
        "right_panel": {"enabled": True, "width": 300}, "rationale": "calm"}


def test_validate_clamps_and_rejects():
    raw = json.loads(json.dumps(GOOD))
    raw["palette"]["wire"] = "not-a-color"
    raw["still_cube"]["edge_style"] = "zigzag"
    raw["bogus"] = 1
    t, warn = vd.validate(raw)
    assert t["palette"]["window_frame"] == "#40e0d0"
    assert t["palette"]["wire"] == vd.DEFAULTS["palette"]["wire"]  # invalid -> default kept
    assert t["still_cube"]["edge_style"] == vd.DEFAULTS["still_cube"]["edge_style"]
    assert t["flame"]["intensity"] == 2.0 and t["flame"]["height"] == 0.3
    assert t["right_panel"]["width"] == 0.24
    assert t["cx_ribbon"]["max_width"] >= t["cx_ribbon"]["base_width"]
    assert "bogus" not in t and warn
    assert vd.validate("nope")[0] == vd.DEFAULTS


def test_schema_covers_defaults():
    assert set(vd.THEME_SCHEMA["required"]) == set(vd.DEFAULTS)
    assert vd.THEME_SCHEMA["properties"]["palette"]["properties"]["badge_fail"]["pattern"]


def test_director_mocked_call_writes_cache(tmp_path):
    fake = FakeClient("```json\n" + json.dumps(GOOD) + "\n```")
    d = vd.Director(cache_path=str(tmp_path / "t.json"), clients={"nebius": fake}).start(background=False)
    assert d.poll()
    assert d.theme["palette"]["cx_link"] == "#ff00ff"
    assert d.label == "theme: muse-spark-open via nebius"
    assert d.info["tokens_in"] == 100 and d.info["cost_usd"] > 0
    assert len(fake.calls) == 1 and fake.calls[0]["response_format"]["type"] == "json_schema"
    saved = json.load(open(tmp_path / "t.json"))
    assert saved["model"] and saved["saved_at"] and saved["theme"]["palette"]["cx_link"] == "#ff00ff"
    # second start uses the cache without calling the model
    fake2 = FakeClient(exc=RuntimeError("should not be called"))
    d2 = vd.Director(cache_path=str(tmp_path / "t.json"), clients={"nebius": fake2}).start(background=False)
    assert "cached" in d2.label and d2.theme["palette"]["cx_link"] == "#ff00ff" and not fake2.calls


def test_error_falls_back_to_cache_then_defaults(tmp_path):
    cache = tmp_path / "t.json"
    d = vd.Director(cache_path=str(cache), clients={"nebius": FakeClient(exc=TimeoutError("slow"))})
    d.start(background=False)
    assert not d.poll() and d.theme == vd.DEFAULTS and d.label == "theme: default" and d.errors
    vd.save_cache(vd.validate(GOOD)[0], {"model": "m/muse", "provider": "fireworks"}, str(cache))
    d = vd.Director(refresh=True, cache_path=str(cache),
                    clients={"nebius": FakeClient(reply="no json here")}).start(background=False)
    assert not d.poll() and d.theme["palette"]["cx_link"] == "#ff00ff" and "cached" in d.label


def test_off_mode_never_calls(tmp_path):
    fake = FakeClient(exc=AssertionError("called"))
    d = vd.Director("off", cache_path=str(tmp_path / "x.json"), clients={"nebius": fake}).start()
    assert d.thread is None and not fake.calls and d.theme == vd.DEFAULTS and "off" in d.label


def test_theme_applied_to_viewer():
    t, _ = vd.validate(GOOD)
    L = viewer.set_look(t, "theme: test")
    try:
        assert L.cx_link == (255, 0, 255) and L.background == (1, 2, 3)
        assert L.ribbon and L.panel and L.still[2] == "dashed"
        assert viewer.scene_w() < viewer.W
        sky = viewer.make_sky(32, 32)
        assert sky.shape == (32, 32, 3)
    finally:
        viewer.set_look(vd.DEFAULTS)
    assert viewer.look().window_frame == viewer.TURQUOISE and viewer.scene_w() == viewer.W
