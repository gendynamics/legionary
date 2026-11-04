# Legionary demo narration (~40 s video)

## Version A: accurate for the current video (deterministic replay, theme by Muse Spark)
| Time | Caption / voiceover |
|---|---|
| 0:00–0:01 | Legionary: an agentic quantum circuit optimizer. |
| 0:02 | A 3-qubit Grover circuit: 157 gates, depth 99. Theme chosen by Muse Spark. |
| 0:03–0:06 | This replay shows the verified reference search that Legionary runs. |
| 0:07–0:09 | Rewrites collapse the circuit: 157, 121, 88, 85 gates. |
| 0:10 | Every candidate is checked against the original. Equivalence: PASS. |
| 0:11–0:28 | The search keeps the best three candidates per round, down to 81 gates. |
| 0:29–0:36 | Done. The Muse Spark agent reached the same 81 gates, driving tools over API. |
| 0:37–0:39 | 157 to 81 gates, depth 99 to 55, proven equivalent. Agent run cost: $0.08. |

## Version B: for a re-recorded agent run (`--planner agent --backend fireworks --viewer`)
| Time | Caption / voiceover |
|---|---|
| 0:00–0:01 | Legionary: Muse Spark as a quantum circuit optimizer agent. |
| 0:02 | Grover, 157 gates. Muse Spark plans; tools do the work. |
| 0:03–0:06 | The agent calls list_strategies, then run_pass, through tool calling. |
| 0:07–0:09 | Each run_pass makes a new circuit: 157, 121, 88, 85 gates. |
| 0:10 | check_equivalence: exact unitary match to the original. PASS. |
| 0:11–0:28 | The agent branches from earlier circuits and reaches 81 gates. |
| 0:29–0:36 | It calls finish. The gate refuses anything not proven equivalent. |
| 0:37–0:39 | 157 to 81, depth 99 to 55. Native on Meta's Responses API. |

Notes: Version B's tool-call lines must match what the re-recorded run actually shows. If B is recorded on Fireworks, that run uses Chat Completions; say "Responses API" only for a `--backend meta` recording. Avoid showing pass names on screen.
