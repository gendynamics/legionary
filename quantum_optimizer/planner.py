"""Muse Spark 1.3 planner via Meta's Responses API (or Baseten)."""
import os, json, sys
from quantum_optimizer import tools
from quantum_optimizer.schemas import TOOLS

ENDPOINTS = {
  "meta":    ("https://api.meta.ai/v1",          "MODEL_API_KEY",   "muse-spark-1.3"),
  "baseten": ("https://inference.baseten.co/v1", "BASETEN_API_KEY", "meta-models/muse-spark-open"),
}


def run(name, args):
    try:
        return getattr(tools, name)(**args)
    except Exception as e:
        return {"error": str(e)}


def client(where):
    from openai import OpenAI
    url, env, _ = ENDPOINTS[where]
    return OpenAI(base_url=url, api_key=os.environ[env])


def plan(prompt, where="meta", turns=8, ai=None):
    # tool_choice is deliberately omitted: muse-spark-1.3 only accepts "auto".
    model = ENDPOINTS[where][2]
    ai = ai or client(where)
    r = ai.responses.create(model=model, input=prompt, tools=TOOLS)
    for _ in range(turns):
        calls = [x for x in r.output if x.type == "function_call"]
        if not calls:
            return r.output_text
        outs = [{"type": "function_call_output",
                 "call_id": c.call_id,
                 "output": json.dumps(run(c.name, json.loads(c.arguments)))}
                for c in calls]
        r = ai.responses.create(model=model, input=outs,
                                previous_response_id=r.id, tools=TOOLS)
    return r.output_text


if __name__ == "__main__":
    where = sys.argv[1] if len(sys.argv) > 1 else "meta"
    qasm = open(sys.argv[2]).read()
    print(plan("Optimize this circuit. Use count_gates, then check_equiv "
               "on every change. QASM:\n" + qasm, where))
