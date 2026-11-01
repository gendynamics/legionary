"""CLI: python -m quantum_optimizer.orchestrator [--planner chain|agent] [--backend auto|meta|fireworks|nebius] [--max-tool-calls N] [--provider meta|baseten|none] [--rounds N] [--beam K] [--only NAME ...] [--[no-]nemotron-review]"""
import argparse, logging, re
from .agents import STRATEGIES, swarm
from .benchmarks import all_benchmarks
from .reviewer import review
from . import nemotron
from .report import table
from .planner import plan
from .rounds import optimize
from . import agent_planner
from . import events
import subprocess, sys


def pick(name, qasm, provider, ai=None, round_no=1, chain=None):
    if provider == "none":
        return list(STRATEGIES)
    prior = (f"Round {round_no}. Strategies already applied in order: {' -> '.join(chain)}. "
             f"Pick what to try NEXT on this already-optimized circuit, best first. ") if chain else ""
    text = plan(f"Circuit '{name}'. {prior}Use count_gates to inspect it, then reply with a "
                f"comma-separated list of strategies to try from: {', '.join(STRATEGIES)}.\n"
                f"QASM:\n{qasm}", where=provider, ai=ai) or ""
    hits = [(m.start(), s) for s in STRATEGIES for m in [re.search(rf"\b{s}\b", text)] if m]
    chosen = [s for _, s in sorted(hits)]  # keep the planner's order
    return chosen or list(STRATEGIES)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["meta", "baseten", "none"], default="none")
    p.add_argument("--nemotron-review", dest="nemotron", action=argparse.BooleanOptionalAction, default=None,
                   help="independent Nemotron review via Nebius (default: on; skipped if NEBIUS_API_KEY unset)")
    p.add_argument("--rounds", type=int, default=4, help="max optimize rounds (stops early at a fixed point)")
    p.add_argument("--beam", type=int, default=3, help="circuits kept per round (1 = greedy)")
    p.add_argument("--planner", choices=["chain", "agent"], default="chain",
                   help="chain = multi-round chaining loop (default); agent = Muse Spark tool-calling planner")
    p.add_argument("--max-tool-calls", type=int, default=agent_planner.DEFAULT_MAX_TOOL_CALLS)
    p.add_argument("--backend", choices=["auto", "meta", "fireworks", "nebius"], default="auto",
                   help="agent planner backend: meta muse-spark-1.3, fireworks/nebius muse-spark-open; auto = meta -> fireworks -> nebius (keyed only)")
    p.add_argument("--only", nargs="*", help="benchmark names to run (default: all)")
    p.add_argument("--events", metavar="PATH|PORT|tcp:PORT",
                   help="emit a live JSONL event stream: a file path, and/or a 127.0.0.1 TCP port (default off)")
    p.add_argument("--viewer", action="store_true",
                   help="open the live 3D viewer window (implies a TCP event stream on a free port)")
    p.add_argument("--director-backend", choices=["auto", "meta", "fireworks", "nebius", "off"], default="auto",
                   help="viewer theme: Muse Spark visual director backend, asked once at viewer startup (off = built-in)")
    p.add_argument("--director-refresh", action="store_true", help="viewer: ignore the cached theme, ask the model again")
    p.add_argument("--renderer", choices=["3d", "2d"], default="3d", help="viewer renderer (default 3d)")
    p.add_argument("--angle", type=float, default=0.0, help="viewer: 3d camera orbit angle offset (degrees)")
    p.add_argument("--render-scale", type=float, default=0.0, help="viewer: 3d internal resolution scale (0 = auto)")
    p.add_argument("--stats", action="store_true", help="viewer: show fps / draw-call stats")
    p.add_argument("--winddown-hold", type=float, default=None, metavar="SECS",
                   help="viewer: seconds removed/failed frames stay visible after the run (default 12)")
    p.add_argument("--winddown-fade", type=float, default=None, metavar="SECS",
                   help="viewer: seconds to fade them out after the hold (default 3)")
    p.add_argument("--no-winddown", action="store_true", help="viewer: never clear removed/failed frames")
    a = p.parse_args(argv)
    vargs = ["--director-backend", a.director_backend, "--renderer", a.renderer, "--angle", str(a.angle)]
    vargs += (["--director-refresh"] if a.director_refresh else []) + (["--stats"] if a.stats else [])
    vargs += ["--render-scale", str(a.render_scale)] if a.render_scale else []
    vargs += ["--winddown-hold", str(a.winddown_hold)] if a.winddown_hold is not None else []
    vargs += ["--winddown-fade", str(a.winddown_fade)] if a.winddown_fade is not None else []
    vargs += ["--no-winddown"] if a.no_winddown else []
    viewer_proc = _setup_events(a.events, a.viewer, vargs)
    lg = logging.getLogger("legionary")
    if not lg.handlers:
        h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    use = nemotron.enabled(a.nemotron)
    second = nemotron.review_rewrite if use else None
    rows, reviews = [], []
    picker = None if a.provider == "none" else (lambda n, q, r, c: pick(n, q, a.provider, round_no=r, chain=c))
    for name, qasm in all_benchmarks().items():
        if a.only and name not in a.only:
            continue
        recs = []
        events.emit_circuit("run_start", qasm, circuit=name, planner=a.planner,
                            max_calls=a.max_tool_calls if a.planner == "agent" else None,
                            max_rounds=a.rounds if a.planner == "chain" else None)
        if a.planner == "agent":
            best = agent_planner.run_agent(name, qasm, max_calls=a.max_tool_calls, backend=a.backend)
            _emit_result(name, best)
            lg.info("agent planner backend ran: %s (%s); tried %s", best["provider"], best["model"], best.get("backends_tried"))
            best["rounds_run"] = best["n_tool_calls"]
            rows.append((name, qasm, best))
            print(f"agent {name}: backend={best['provider']} model={best['model']} api={best['api']} "
                  f"chain={best['strategy']} source={best['source']} nudged={best['nudged']} equiv={best['equiv']} "
                  f"tool_calls={best['n_tool_calls']} tokens={best['usage']} cost=${best['cost_usd']:.5f}")
            for i, c in enumerate(best["tool_calls"], 1):
                print(f"  call {i}: {c['name']}({c['args']}) -> {c['result']}")
            for e in best["errors"]:
                print(f"  API error: {e}")
            continue
        best = optimize(name, qasm, picker=picker, rounds=a.rounds, beam=a.beam, second_opinion=second, records=recs)
        _emit_result(name, best)
        rows.append((name, qasm, best))
        reviews += [(name, r) for r in recs]
    print(table(rows))
    if use:
        print(review_table(reviews))
    for name, _, best in rows:
        print(f"chain {name}: {best['strategy']} ({best['rounds_run']} rounds)")
    events.emit("run_end")
    events.close()
    if viewer_proc is not None:
        print("viewer still open; close its window (or press Esc) when done")


def _setup_events(spec, viewer, viewer_args=()):
    path = port = None
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("tcp:"):
            port = int(part[4:])
        elif part.isdigit():
            port = int(part)
        else:
            path = part
    if viewer and port is None:
        port = 0  # free port
    if path is None and port is None:
        return None
    em = events.configure(path=path, port=port)
    if port is not None:
        print(f"event stream: tcp 127.0.0.1:{em.port}")
    if path:
        print(f"event stream: {path}")
    if not viewer:
        return None
    import time
    proc = subprocess.Popen([sys.executable, "-m", "quantum_optimizer.viewer", "--connect", f"127.0.0.1:{em.port}",
                             *viewer_args])
    time.sleep(1.5)  # let the window open; backlog replay covers anything missed
    return proc


def _emit_result(name, best):
    events.emit_circuit("circuit_update", best["qasm"], circuit_id="final", chain=best.get("strategy"))
    events.emit("result", circuit=name, source=best.get("source", "chain"), strategy=best.get("strategy"),
                gates=best.get("gates", best.get("total")), depth=best.get("depth"), cx=best.get("cx"),
                equiv=best.get("equiv"), tokens=best.get("usage"), cost_usd=best.get("cost_usd"))


def review_table(reviews):
    out = ["", "| circuit | rewrite | nemotron status | nemotron verdict | equiv |", "|---|---|---|---|---|"]
    for name, r in reviews:
        n = r.get("nemotron", {})
        out.append(f"| {name} | {r['strategy']} | {n.get('status')} | {n.get('verdict') or '-'} | "
                   f"{'yes' if r['equiv'] else 'NO'} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
