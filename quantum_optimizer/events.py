"""Structured live event stream (JSON lines). Off by default: emit() is a no-op until configure().

Sinks: a JSONL file path and/or a TCP broadcast server on 127.0.0.1:<port>. New TCP clients get the
backlog of the current run replayed first, so a viewer can connect late. Never carries secrets:
only circuit structure, metrics, tool names/args, token counts and cost.

Event types: run_start, tool_call, circuit_update, equivalence, result, run_end.
"""
import json, socket, threading, time

_sink = None


def circuit_gates(qasm):
    """Gate list with ASAP layer positions: [{name, qubits, layer}], plus n_qubits."""
    from .tools import load
    qc = load(qasm)
    level = [0] * qc.num_qubits
    out = []
    for inst in qc.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if not qs:
            continue
        lay = max(level[q] for q in qs)
        for q in qs:
            level[q] = lay + 1
        out.append({"name": inst.operation.name, "qubits": qs, "layer": lay})
    return out, qc.num_qubits


def snapshot(qasm):
    from .tools import count_gates
    m = count_gates(qasm)
    gates, nq = circuit_gates(qasm)
    return {"gates": m["total"], "depth": m["depth"], "cx": m["cx"], "n_qubits": nq, "gate_list": gates}


class Emitter:
    def __init__(self, path=None, port=None):
        self.lock = threading.Lock()
        self.f = open(path, "a", buffering=1) if path else None
        self.clients, self.backlog, self.server = [], [], None
        if port is not None:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(("127.0.0.1", int(port)))
            self.server.listen(4)
            self.port = self.server.getsockname()[1]
            threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                c, _ = self.server.accept()
            except OSError:
                return
            with self.lock:
                try:
                    for line in self.backlog:
                        c.sendall(line)
                    self.clients.append(c)
                except OSError:
                    c.close()

    def emit(self, ev):
        line = (json.dumps(ev, default=str) + "\n").encode()
        with self.lock:
            if self.f:
                self.f.write(line.decode())
            if self.server:
                if ev.get("type") == "run_start":
                    self.backlog = []
                self.backlog.append(line)
                for c in list(self.clients):
                    try:
                        c.sendall(line)
                    except OSError:
                        self.clients.remove(c)

    def close(self):
        with self.lock:
            if self.f:
                self.f.close()
            for c in self.clients:
                try:
                    c.close()
                except OSError:
                    pass
            if self.server:
                self.server.close()


def configure(path=None, port=None):
    """Enable the stream. Returns the Emitter (has .port when TCP is on)."""
    global _sink
    close()
    _sink = Emitter(path, port)
    return _sink


def close():
    global _sink
    if _sink:
        _sink.close()
    _sink = None


def enabled():
    return _sink is not None


def emit(type_, **data):
    if _sink is None:
        return
    try:
        _sink.emit({"type": type_, "t": round(time.time(), 3), **data})
    except Exception:
        pass  # the stream must never break an optimization run


def emit_circuit(type_, qasm, **data):
    if _sink is None:
        return
    try:
        snap = snapshot(qasm)
    except Exception:
        return
    emit(type_, **snap, **data)
