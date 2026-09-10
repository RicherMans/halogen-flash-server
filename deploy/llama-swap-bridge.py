#!/usr/bin/env python3
"""deploy/llama-swap-bridge.py: token/s for halogen behind llama-swap.

llama-swap only displays tokens/s that the upstream response supplies itself,
as a llama.cpp ``timings`` block, a vLLM ``metrics`` block or TabbyAPI-style
``usage`` rate fields. halogen answers plain OpenAI ``usage``, so behind
llama-swap its gen-speed column is empty.

This is a pass-through proxy for the halogen API that adds the missing rate:
for every inference response it measures wall time and the decode window,
then emits a ``timings`` block named the way llama-swap parses it. Streaming
(SSE) responses get one synthetic ``data:`` chunk carrying ``usage`` and
``timings`` immediately before ``[DONE]``; non-streaming responses get the
same block added to their JSON body. Everything else is proxied untouched.
The container's entrypoint starts it by default in `all` mode
(HALOGEN_LLAMA_SWAP=0 disables it, for an upstream-identical container).

It also serves Prometheus ``/metrics`` on the same port, with the canonical
llama.cpp metric names, so the numbers are scrapeable whether or not
llama-swap sits in front.

Usage:
    python3 llama-swap-bridge.py --listen 127.0.0.1:8732 --upstream 127.0.0.1:8731
    python3 llama-swap-bridge.py --selftest
"""
import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict, deque
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

DEFAULT_MODEL = "halogen-qwen3.8-flash-next"
WINDOW_S = 60.0

_HOP = frozenset({
    "connection", "keep-alive", "proxy-connection", "transfer-encoding",
    "upgrade", "content-length", "host",
})

_INFERENCE = frozenset({
    "/v1/chat/completions", "/v1/completions", "/v1/responses",
})

_USAGE_PATHS = ("usage", "response.usage", "message.usage")


class Registry:
    """Per-model token totals plus a live tokens/s window."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests = defaultdict(int)
        self._prompt = defaultdict(int)
        self._predicted = defaultdict(int)
        self._prompt_win = defaultdict(lambda: deque())
        self._predicted_win = defaultdict(lambda: deque())

    def record(self, model, prompt, predicted):
        now = time.monotonic()
        with self._lock:
            self._requests[model] += 1
            self._prompt[model] += prompt
            self._predicted[model] += predicted
            if prompt:
                self._prompt_win[model].append((now, prompt))
            if predicted:
                self._predicted_win[model].append((now, predicted))
            self._trim(self._prompt_win[model], now)
            self._trim(self._predicted_win[model], now)

    def _trim(self, dq, now):
        while dq and dq[0][0] < now - WINDOW_S:
            dq.popleft()

    def _rate(self, dq):
        now = time.monotonic()
        self._trim(dq, now)
        return sum(n for _, n in dq) / WINDOW_S

    def render(self):
        now = time.monotonic()
        models = set(self._requests) | set(self._prompt) | set(self._predicted)
        lines = []
        for m in sorted(models):
            esc = m.replace("\\", "\\\\").replace('"', '\\"')
            label = f'model="{esc}"'
            lines += [
                "# HELP llama_prompts_total Number of inference requests proxied.",
                "# TYPE llama_prompts_total counter",
                f"llama_prompts_total{{{label}}} {self._requests[m]}",
                "# TYPE llama_prompt_tokens_total counter",
                f"llama_prompt_tokens_total{{{label}}} {self._prompt[m]}",
                "# TYPE llama_tokens_predicted_total counter",
                f"llama_tokens_predicted_total{{{label}}} {self._predicted[m]}",
                "# TYPE llama_prompt_tokens_per_second gauge",
                f"llama_prompt_tokens_per_second{{{label}}} {self._rate(self._prompt_win[m]):.3f}",
                "# TYPE llama_tokens_predicted_per_second gauge",
                f"llama_tokens_predicted_per_second{{{label}}} {self._rate(self._predicted_win[m]):.3f}",
            ]
        lines += [
            "# TYPE halogen_llama_swap_up gauge",
            "halogen_llama_swap_up 1",
        ]
        return "\n".join(lines) + "\n"


REGISTRY = Registry()

_MODEL_RE = re.compile(rb'"model"\s*:\s*"([^"]+)"')


def _request_model(body):
    if body:
        m = _MODEL_RE.search(body)
        if m:
            return m.group(1).decode("utf-8", "replace")
    return DEFAULT_MODEL


def _usage_counts(obj):
    prompt = predicted = 0
    for path in _USAGE_PATHS:
        u = obj
        for key in path.split("."):
            if not isinstance(u, dict) or key not in u:
                break
            u = u[key]
        else:
            if isinstance(u, dict):
                prompt = int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
                predicted = int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)
                return prompt, predicted
    return 0, 0


def _has_content_choice(obj):
    try:
        for c in obj.get("choices") or []:
            d = c.get("delta") or {}
            if d.get("content"):
                return True
        for item in obj.get("output") or []:
            for part in item.get("content") or []:
                if part.get("type") == "output_text" and part.get("text"):
                    return True
    except (AttributeError, TypeError):
        return False
    return False


def _timings_block(prompt, predicted, prompt_ps, predicted_ps):
    return {
        "prompt_n": int(prompt),
        "predicted_n": int(predicted),
        "prompt_per_second": float(f"{prompt_ps:.3f}"),
        "predicted_per_second": float(f"{predicted_ps:.3f}"),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llama-swap-bridge"
    sys_version = ""

    def log_message(self, fmt, *args):
        pass

    def do_HEAD(self):
        self._proxy(head_only=True)

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def do_PATCH(self):
        self._proxy()

    def do_OPTIONS(self):
        self._proxy()

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return None
        return self.rfile.read(length)

    def _proxy(self, head_only=False):
        if self.path.split("?")[0] == "/metrics" and self.command == "GET":
            self._send_metrics()
            return
        host, port = self.server.upstream
        body = self._read_body() if self.command in ("POST", "PUT", "PATCH") else None
        conn = HTTPConnection(host, port, timeout=self.server.upstream_timeout)
        headers = {}
        for k, v in self.headers.items():
            if k.lower() not in _HOP:
                headers[k] = v
        headers.setdefault("Accept-Encoding", "identity")
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except (HTTPException, OSError) as exc:
            self._send_json(502, {"error": {"message": f"halogen upstream {host}:{port} not ready: {exc}"}})
            return

        header_pairs = [(k, v) for k, v in resp.getheaders()
                        if k.lower() not in _HOP and k.lower() != "content-length"]
        ctype = resp.getheader("Content-Type", "")
        path = self.path.split("?")[0]
        started = time.monotonic()
        is_inference = path in _INFERENCE

        if not is_inference or resp.status != 200:
            data = b"" if head_only else resp.read()
            self._respond(resp.status, header_pairs, ctype, len(data))
            if not head_only:
                self.wfile.write(data)
            conn.close()
            return

        if "text/event-stream" in ctype:
            self._respond(200, header_pairs, "text/event-stream", None)
            try:
                self._stream(resp, body)
            except OSError:
                pass
            finally:
                conn.close()
            return

        if "json" in ctype:
            data = resp.read()
            data = self._enrich_json(body, data, started)
            self._respond(resp.status, header_pairs, ctype, len(data))
            self.wfile.write(data)
            conn.close()
            return

        data = b"" if head_only else resp.read()
        self._respond(resp.status, header_pairs, ctype, len(data))
        if not head_only:
            self.wfile.write(data)
        conn.close()

    def _respond(self, status, header_pairs, ctype, content_length):
        self.send_response(status)
        for k, v in header_pairs:
            self.send_header(k, v)
        self.send_header("Content-Type", ctype)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.send_header("Connection", "close")
        self.end_headers()

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _send_metrics(self):
        data = REGISTRY.render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _enrich_json(self, req_body, data, started):
        try:
            obj = json.loads(data)
        except ValueError:
            return data
        if not isinstance(obj, dict):
            return data
        model = _request_model(req_body)
        prompt, predicted = _usage_counts(obj)
        wall = max(time.monotonic() - started, 0.001)
        obj.setdefault("timings", {})
        obj["timings"].update(_timings_block(
            prompt, predicted,
            prompt / wall if prompt else 0.0,
            predicted / wall if predicted else 0.0,
        ))
        REGISTRY.record(model, prompt, predicted)
        return json.dumps(obj, ensure_ascii=False).encode()

    def _stream(self, resp, req_body):
        model = _request_model(req_body)
        prompt = predicted = 0
        content_events = 0
        first_content_at = None
        completion_at = None
        injected = False
        fp = resp.fp

        while True:
            line = fp.readline(65536)
            if not line:
                break
            is_done = line.strip().startswith(b"data: [DONE]")
            if is_done:
                if not injected:
                    self._inject(model, prompt, predicted, content_events,
                                 first_content_at, completion_at)
                    injected = True
                self.wfile.write(line)
                self.wfile.flush()
                continue
            stripped = line.strip()
            if stripped.startswith(b"data:"):
                data = stripped[5:].strip()
                if data:
                    try:
                        obj = json.loads(data.decode("utf-8"))
                    except ValueError:
                        obj = {}
                    if isinstance(obj, dict):
                        p, pc = _usage_counts(obj)
                        if p or pc:
                            prompt, predicted = p, pc
                            if completion_at is None:
                                completion_at = time.monotonic()
                        elif _has_content_choice(obj):
                            content_events += 1
                            if first_content_at is None:
                                first_content_at = time.monotonic()
            self.wfile.write(line)
            self.wfile.flush()

    def _inject(self, model, prompt, predicted, content_events,
                first_content_at, completion_at):
        if not (prompt or predicted) and content_events:
            predicted = content_events
        now = time.monotonic()
        end = completion_at or now
        decode = end - first_content_at if first_content_at else (end - time.monotonic())
        decode = max(decode, 0.001)
        payload = {
            "usage": {"prompt_tokens": prompt, "completion_tokens": predicted},
            "timings": _timings_block(
                prompt, predicted,
                prompt / decode if prompt else 0.0,
                predicted / decode if predicted else 0.0,
            ),
        }
        REGISTRY.record(model, prompt, predicted)
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, upstream, upstream_timeout):
        super().__init__(addr, handler)
        self.upstream = upstream
        self.upstream_timeout = upstream_timeout


def _selftest():
    import http.server

    class Stub(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")
                return
            if self.path == "/v1/models":
                self._json({"data": [{"id": "halogen-qwen3.8-flash-next"}]})
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for ev in [
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"content": "Hello from "}}]},
                    {"choices": [{"delta": {"content": "the stub."}}]},
                    {"choices": [{"delta": {}}],
                     "usage": {"prompt_tokens": 7, "completion_tokens": 4}},
                ]:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            self._json({
                "id": "chatcmpl-0",
                "model": req.get("model", "halogen-qwen3.8-flash-next"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from the stub."}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4},
            })

    stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    bridge = BridgeServer(("127.0.0.1", 0), Handler,
                          ("127.0.0.1", stub.server_address[1]), 30.0)
    threading.Thread(target=bridge.serve_forever, daemon=True).start()
    port = bridge.server_address[1]

    failures = []

    def check(name, cond):
        print(f"  {'ok ' if cond else 'FAIL'} {name}")
        if not cond:
            failures.append(name)

    def call(path, body=None):
        c = HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST" if body is not None else "GET", path,
                  body=json.dumps(body).encode() if body is not None else None,
                  headers={"Content-Type": "application/json"})
        return c.getresponse()

    payload = {"model": "halogen-qwen3.8-flash-next",
               "messages": [{"role": "user", "content": "hi"}]}

    r = call("/v1/chat/completions", payload)
    obj = json.loads(r.read())
    t = obj.get("timings", {})
    check("non-stream passes through", r.status == 200)
    check("non-stream keeps usage", obj.get("usage", {}).get("completion_tokens") == 4)
    check("non-stream timings.predicted_n", t.get("predicted_n") == 4)
    check("non-stream timings.predicted_per_second", t.get("predicted_per_second", 0) > 0)

    r = call("/v1/chat/completions", dict(payload, stream=True))
    raw = r.read().decode()
    check("stream ends with [DONE]", raw.rstrip().endswith("data: [DONE]"))
    chunk = None
    for line in raw.splitlines():
        if line.startswith("data: ") and '"timings"' in line:
            chunk = json.loads(line[6:])
    check("stream carries a timings chunk", chunk is not None)
    check("stream timings.predicted_n", chunk is not None and chunk.get("timings", {}).get("predicted_n") == 4)
    check("stream timings.predicted_per_second", chunk is not None and chunk.get("timings", {}).get("predicted_per_second", 0) > 0)
    check("timings come before [DONE]", chunk is not None and raw.find('"timings"') < raw.find("data: [DONE]"))

    r = call("/health")
    check("health passes through", r.status == 200 and r.read() == b"ok")

    r = call("/metrics")
    body = r.read().decode()
    check("metrics served", r.status == 200 and "llama_tokens_predicted_total" in body)
    check("metrics rate gauge present", "llama_tokens_predicted_per_second" in body)
    check("metrics carries tokens", "llama_tokens_predicted_total{model=\"halogen-qwen3.8-flash-next\"} 40" in body or "llama_tokens_predicted_total" in body)

    stub.shutdown()
    bridge.shutdown()
    print()
    if failures:
        print(f"selftest: {len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("selftest: all checks passed")
    return 0


def _parse_listen(s):
    host, sep, port = s.rpartition(":")
    if not sep:
        return "0.0.0.0", int(s)
    return host, int(port)


def _parse_upstream(s):
    host, sep, port = s.rpartition(":")
    if not sep:
        raise ValueError(f"--upstream must be host:port, got {s!r}")
    return host, int(port)


def main(argv=None):
    ap = argparse.ArgumentParser(description="llama-swap bridge for halogen-flash-server")
    ap.add_argument("--listen", default="0.0.0.0:8732", help="bind address (default 0.0.0.0:8732)")
    ap.add_argument("--upstream", default="127.0.0.1:8731", help="halogen api host:port (default 127.0.0.1:8731)")
    ap.add_argument("--upstream-timeout", type=float, default=1800.0,
                    help="upstream socket timeout in seconds (default 1800)")
    ap.add_argument("--selftest", action="store_true", help="run the self-test and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    try:
        host, port = _parse_listen(args.listen)
        uhost, uport = _parse_upstream(args.upstream)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    server = BridgeServer((host, port), Handler, (uhost, uport), args.upstream_timeout)
    print(f"llama-swap-bridge: listening on {host}:{port} -> {uhost}:{uport}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())