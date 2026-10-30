"""Tool schemas in the flat format used by Meta's Responses API."""
TOOLS = [
  {
    "type": "function",
    "name": "count_gates",
    "description": "Count gates and depth of a circuit.",
    "parameters": {
      "type": "object",
      "properties": {
        "qasm": {"type": "string", "description": "Circuit in OpenQASM 2."}
      },
      "required": ["qasm"],
      "additionalProperties": False
    }
  },
  {
    "type": "function",
    "name": "check_equiv",
    "description": "Check that two circuits do the same thing.",
    "parameters": {
      "type": "object",
      "properties": {
        "before": {"type": "string", "description": "Original circuit in QASM."},
        "after":  {"type": "string", "description": "Optimized circuit in QASM."},
        "tol":    {"type": "number", "description": "Tolerance, like 1e-8."}
      },
      "required": ["before", "after"],
      "additionalProperties": False
    }
  },
  {
    "type": "function",
    "name": "simulate",
    "description": "Run a circuit on a simulator and return counts.",
    "parameters": {
      "type": "object",
      "properties": {
        "qasm":  {"type": "string",  "description": "Circuit in QASM."},
        "shots": {"type": "integer", "description": "Number of shots, like 1024."},
        "seed":  {"type": "integer", "description": "Random seed for repeat runs."}
      },
      "required": ["qasm"],
      "additionalProperties": False
    }
  }
]
