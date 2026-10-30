"""Independent second-opinion reviewer: NVIDIA Nemotron 3 Super 120B on Nebius Token Factory.

Advisory only. It never replaces the exact check_equiv test, which stays the authority.
Uses the OpenAI-compatible chat completions API (not the Responses API).
"""
import json, os, re, logging

NEBIUS = ("https://api.tokenfactory.nebius.com/v1", "NEBIUS_API_KEY", "nvidia/nemotron-3-super-120b-a12b")
VERDICTS = {"approve", "reject", "uncertain"}
log = logging.getLogger("legionary.nemotron")

PROMPT = ("You are an independent reviewer of quantum circuit rewrites. Compare BEFORE and AFTER "
          "OpenQASM 2 circuits and judge whether AFTER is a correct, equivalent optimization. "
          "Reply with ONLY a JSON object: {\"verdict\": \"approve\"|\"reject\"|\"uncertain\", "
          "\"reasons\": [strings], \"risk_notes\": [strings]}.")


def enabled(flag=None):
    """flag None = auto: on when NEBIUS_API_KEY is set, unless LEGIONARY_NEMOTRON_REVIEW=0."""
    if flag is None:
        flag = os.environ.get("LEGIONARY_NEMOTRON_REVIEW", "1").lower() not in ("0", "false", "no", "off")
    return bool(flag)


def client():
    from openai import OpenAI
    url, env, _ = NEBIUS
    return OpenAI(base_url=url, api_key=os.environ[env], timeout=60)


def parse(text):
    """Defensive JSON parse: tolerates code fences, extra prose, bad fields."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    v = str(d.get("verdict", "")).strip().lower()
    as_list = lambda x: [str(i) for i in x] if isinstance(x, list) else ([str(x)] if x else [])
    return {"verdict": v if v in VERDICTS else "uncertain",
            "reasons": as_list(d.get("reasons")), "risk_notes": as_list(d.get("risk_notes"))}


def review_rewrite(before, after, strategy="", ai=None):
    """Return {"status": ok|skipped|error, "verdict", "reasons", "risk_notes", ...}. Never raises."""
    rec = {"model": NEBIUS[2], "status": "skipped", "verdict": None, "reasons": [], "risk_notes": []}
    if ai is None and not os.environ.get(NEBIUS[1]):
        rec["error"] = f"{NEBIUS[1]} not set"
        return rec
    try:
        ai = ai or client()
        r = ai.chat.completions.create(
            model=NEBIUS[2], temperature=0, max_tokens=600,
            messages=[{"role": "system", "content": PROMPT},
                      {"role": "user", "content": f"Strategy: {strategy}\nBEFORE:\n{before}\nAFTER:\n{after}"}])
        parsed = parse(r.choices[0].message.content)
        if parsed is None:
            rec.update(status="error", error="malformed JSON from reviewer model")
        else:
            rec.update(status="ok", **parsed)
    except Exception as e:  # network/auth/rate-limit: advisory step, never crash the run
        rec.update(status="error", error=type(e).__name__)
    return rec
