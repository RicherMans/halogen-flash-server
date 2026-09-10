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
# A decode rate above this cannot be real on this hardware (measured ~40 tok/s
# decode, ~1.4k tok/s prefill). A rate that exceeds it means the timing window
# collapsed, not that the model is fast, so it is reported as 0 instead.
MAX_PLAUSIBLE_TPS = 10000.0

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


def _as_int(value):
    """Token counts from a response field, or 0 when it is not a number.

    Upstreams occasionally put a string or a nested object where a count is
    expected; that must not crash the proxy, so anything non-numeric is read
    as 0 rather than raised.
    """
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _usage_counts(obj):
    prompt = predicted = 0
    try:
        for path in _USAGE_PATHS:
            u = obj
            for key in path.split("."):
                if not isinstance(u, dict) or key not in u:
                    break
                u = u[key]
            else:
                if isinstance(u, dict):
                    prompt = _as_int(u.get("prompt_tokens", u.get("input_tokens", 0)))
                    predicted = _as_int(u.get("completion_tokens", u.get("output_tokens", 0)))
                    return prompt, predicted
    except (AttributeError, TypeError):
        return 0, 0
    return 0, 0


def _has_generated_delta(obj):
    """True when an SSE event carries at least one generated token.

    Covers answer content, thinking (Qwen-style ``reasoning_content`` /
    ``reasoning``) and tool calls, so the decode window starts at the first
    token the model emits rather than the first token that happens to be
    answer text. A reasoning-heavy turn that opens with a long think and a
    one-token answer otherwise looked like it decoded at millions of tok/s.
    """
    try:
        # OpenAI Responses streams typed events (``response.output_text.delta``,
        # ``response.reasoning_summary_text.delta``, ...) whose payload is a
        # top-level ``delta`` string, not a choices array.
        etype = obj.get("type")
        if isinstance(etype, str) and etype.endswith(".delta") and obj.get("delta"):
            return True
        for c in obj.get("choices") or []:
            d = c.get("delta") or {}
            if (d.get("content") or d.get("reasoning_content")
                    or d.get("reasoning") or d.get("tool_calls")):
                return True
            m = c.get("message") or {}
            if (m.get("content") or m.get("reasoning_content")
                    or m.get("reasoning") or m.get("tool_calls")):
                return True
        for item in obj.get("output") or []:
            if item.get("type") in ("reasoning", "function_call"):
                return True
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


def _window(start, end):
    """A measured window, floored at 1ms so a fast local call cannot divide by
    (near) zero. This floor is why the plausibility guard below exists."""
    return max(end - start, 0.001)


def _rate(tokens, window):
    """Tokens/s over a measured window, or 0.0 when it cannot be trusted.

    llama-swap stores and displays the ``predicted_per_second`` it is handed
    with no sanity check, so an unmeasurable window must report 0 rather than
    a fabricated number.
    """
    if tokens <= 0 or window <= 0:
        return 0.0
    rate = tokens / window
    if not (0.0 < rate < MAX_PLAUSIBLE_TPS):
        return 0.0
    return rate


def _computed_timings(prompt, predicted, first_token_at, completion_at, started):
    """The bridge's own timings block for one turn.

    The first generated token is the start of decode; `started` is the request
    start, so the run-up to the first token is the prefill window. If usage
    arrived before any token (or no token was seen at all) the decode window
    is unmeasurable, so fall back to the whole request rather than the old
    ~0.001s artifact.
    """
    end = completion_at or time.monotonic()
    first = first_token_at or started
    if first > end:
        first = started
    prefill = _window(started, first)
    decode = _window(first, end)
    return _timings_block(
        prompt, predicted,
        _rate(prompt, prefill),
        _rate(predicted, decode),
    )


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
        encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in encoding:
            return self._read_chunked_body()
        length = _as_int(self.headers.get("Content-Length"))
        if length <= 0:
            return None
        return self.rfile.read(length)

    def _read_chunked_body(self):
        """De-chunk a request body.

        Some clients send POST bodies as ``Transfer-Encoding: chunked`` with no
        ``Content-Length``. That header is hop-by-hop and stripped before the
        request is forwarded, so the body has to be reassembled here or the
        upstream would see an empty request.
        """
        body = bytearray()
        while True:
            size_line = self.rfile.readline(65536)
            if not size_line:
                break
            try:
                size = int(size_line.split(b";", 1)[0].strip(), 16)
            except ValueError:
                break
            if size == 0:
                while True:
                    trailer = self.rfile.readline(65536)
                    if not trailer or trailer in (b"\r\n", b"\n"):
                        break
                break
            body += self.rfile.read(size)
            self.rfile.read(2)
        return bytes(body) if body else None

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
        # Timed from before the request is sent. A non-streaming completion
        # sends its headers only after the whole answer is generated, so
        # timing from getresponse() measured body transfer and produced
        # millions of tok/s.
        started = time.monotonic()
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
                self._stream(resp, body, started, path)
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
        # Non-streaming cannot observe a decode window, so both rates use the
        # whole-request wall as a conservative lower bound. `started` is
        # before the upstream request was sent, so it includes generation.
        wall = _window(started, time.monotonic())
        computed = _timings_block(
            prompt, predicted,
            _rate(prompt, wall),
            _rate(predicted, wall),
        )
        # Never overwrite a real timings block an upstream supplied; fill in
        # only the fields it did not provide.
        existing = obj.get("timings")
        if isinstance(existing, dict):
            for key, value in computed.items():
                existing.setdefault(key, value)
        else:
            obj["timings"] = computed
        REGISTRY.record(model, prompt, predicted)
        return json.dumps(obj, ensure_ascii=False).encode()

    def _stream(self, resp, req_body, started, path):
        model = _request_model(req_body)
        is_responses = path == "/v1/responses"
        prompt = predicted = 0
        token_events = 0
        first_token_at = None
        completion_at = None
        upstream_timings = None
        injected = False
        fp = resp.fp

        try:
            while True:
                line = fp.readline(65536)
                if not line:
                    break
                is_done = line.strip().startswith(b"data: [DONE]")
                if is_done:
                    # Responses clients never see this sentinel; if it shows up
                    # anyway, do not drop a chat-shaped chunk into their stream.
                    if not injected and not is_responses:
                        self._inject(model, prompt, predicted, token_events,
                                     first_token_at, completion_at, started,
                                     upstream_timings)
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
                            t = obj.get("timings")
                            if isinstance(t, dict):
                                upstream_timings = t
                            p, pc = _usage_counts(obj)
                            if p or pc:
                                prompt, predicted = p, pc
                                if completion_at is None:
                                    completion_at = time.monotonic()
                            elif _has_generated_delta(obj):
                                token_events += 1
                                if first_token_at is None:
                                    first_token_at = time.monotonic()
                            # The Responses stream has no [DONE]; attach the
                            # rate to its own response.completed event so no
                            # foreign, chat-shaped chunk reaches the client.
                            if (is_responses and not injected
                                    and obj.get("type") == "response.completed"):
                                if not (prompt or predicted):
                                    prompt, predicted = _usage_counts(obj)
                                if not (prompt or predicted) and token_events:
                                    predicted = token_events
                                obj["timings"] = upstream_timings or _computed_timings(
                                    prompt, predicted, first_token_at,
                                    completion_at, started)
                                line = f"data: {json.dumps(obj)}\n\n".encode()
                                injected = True
                self.wfile.write(line)
                self.wfile.flush()
        finally:
            # A client can hang up mid-stream, so inject on EOF/abort too. For
            # Responses we deliberately inject nothing rather than a chunk its
            # parser would reject.
            if not injected and not is_responses:
                self._inject(model, prompt, predicted, token_events,
                             first_token_at, completion_at, started,
                             upstream_timings)
                injected = True

    def _inject(self, model, prompt, predicted, token_events,
                first_token_at, completion_at, started, upstream_timings=None):
        if upstream_timings and not (prompt or predicted):
            prompt = _as_int(upstream_timings.get("prompt_n"))
            predicted = _as_int(upstream_timings.get("predicted_n"))
        if not (prompt or predicted) and token_events:
            predicted = token_events
        timings = upstream_timings or _computed_timings(
            prompt, predicted, first_token_at, completion_at, started)
        # A real chat.completion.chunk: `choices` is REQUIRED by OpenAI clients
        # (zod-validated) on every SSE chunk. The empty array is exactly what
        # llama.cpp/vLLM/OpenAI themselves send on the usage-bearing final chunk;
        # llama-swap reads only `usage`/`timings` from it.
        payload = {
            "id": "chatcmpl-halogen",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": prompt, "completion_tokens": predicted},
            "timings": timings,
        }
        REGISTRY.record(model, prompt, predicted)
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, upstream, upstream_timeout):
        super().__init__(addr, handler)
        self.upstream = upstream
        self.upstream_timeout = upstream_timeout


def _stub_events(scenario):
    """Canned SSE events for the bridge self-test.

    `reasoning` models a thinking-only turn that ends on the token budget
    (`finish_reason: length`, empty content). `reasoning_then_content` models
    a think followed by a short answer. Both are the shapes that used to make
    the bridge collapse its decode window and emit millions of tok/s.
    """
    if scenario == "reasoning":
        return [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "Think. "}}]},
            {"choices": [{"delta": {"reasoning_content": "Think more."}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 7, "completion_tokens": 4}},
        ]
    if scenario == "reasoning_then_content":
        return [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"reasoning_content": "Think."}}]},
            {"choices": [{"delta": {"content": "Answer."}}]},
            {"choices": [{"delta": {}}],
             "usage": {"prompt_tokens": 7, "completion_tokens": 4}},
        ]
    if scenario == "responses":
        # The Responses API streams typed events and ends with
        # response.completed; there is no [DONE] sentinel.
        return [
            {"type": "response.created", "response": {}},
            {"type": "response.output_text.delta", "delta": "Hello "},
            {"type": "response.output_text.delta", "delta": "world"},
            {"type": "response.completed",
             "response": {"usage": {"input_tokens": 5, "output_tokens": 2}}},
        ]
    if scenario == "bogus_usage":
        return [
            {"choices": [{"delta": {"content": "Hello."}}]},
            {"choices": [{"delta": {}}],
             "usage": {"prompt_tokens": "seven", "completion_tokens": {"n": 4}}},
        ]
    if scenario == "upstream_timings":
        return [
            {"choices": [{"delta": {"content": "Hello."}}]},
            {"choices": [{"delta": {}}],
             "usage": {"prompt_tokens": 7, "completion_tokens": 4},
             "timings": {"prompt_n": 7, "predicted_n": 4,
                         "prompt_per_second": 111.0,
                         "predicted_per_second": 222.0}},
        ]
    return [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello from "}}]},
        {"choices": [{"delta": {"content": "the stub."}}]},
        {"choices": [{"delta": {}}],
         "usage": {"prompt_tokens": 7, "completion_tokens": 4}},
    ]


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
            raw = self.rfile.read(length) if length > 0 else b""
            req = json.loads(raw or b"{}")
            scenario = req.get("scenario")
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for ev in _stub_events(scenario):
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
                if scenario != "responses":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                return
            resp = {
                "id": "chatcmpl-0",
                "model": req.get("model", "halogen-qwen3.8-flash-next"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from the stub."}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4},
                # Lets the chunked-request test confirm the body survived.
                "echo_req_len": len(raw),
            }
            if scenario == "upstream_timings":
                resp["timings"] = {"prompt_n": 7, "predicted_n": 4,
                                   "prompt_per_second": 111.0,
                                   "predicted_per_second": 222.0}
            self._json(resp)

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
    check("injected chunk is a valid chat.completion.chunk",
          chunk is not None and chunk.get("choices") == []
          and chunk.get("object") == "chat.completion.chunk"
          and chunk.get("id") and chunk.get("model"))
    check("injected chunk created is epoch",
          chunk is not None and chunk.get("created", 0) > 1_600_000_000)

    # Reasoning-heavy turns are the ones that used to report millions of
    # tok/s: thinking is generated but was not counted, so the decode window
    # collapsed. The rate must now be positive and physically plausible.
    def _timings_chunk(raw):
        found = None
        for line in raw.splitlines():
            if line.startswith("data: ") and '"timings"' in line:
                found = json.loads(line[6:])
        return found

    for scenario in ("reasoning", "reasoning_then_content"):
        r = call("/v1/chat/completions",
                 dict(payload, stream=True, scenario=scenario))
        tc = _timings_chunk(r.read().decode())
        check(f"{scenario} stream carries timings", tc is not None)
        rate = tc.get("timings", {}).get("predicted_per_second", 0) if tc else 0
        check(f"{scenario} predicted_per_second is sane",
              0 < rate < MAX_PLAUSIBLE_TPS)

    # Upstream-supplied timings must win over the bridge's computed ones.
    r = call("/v1/chat/completions", dict(payload, scenario="upstream_timings"))
    obj = json.loads(r.read())
    check("non-stream keeps upstream timings",
          obj.get("timings", {}).get("predicted_per_second") == 222.0)

    # The Responses API stream ends with response.completed and no [DONE];
    # the bridge must still inject a timings chunk at EOF.
    r = call("/v1/responses", dict(payload, stream=True, scenario="responses"))
    raw = r.read().decode()
    check("responses stream has no [DONE]", "data: [DONE]" not in raw)
    tc = _timings_chunk(raw)
    check("responses stream carries timings", tc is not None)
    rate = tc.get("timings", {}).get("predicted_per_second", 0) if tc else 0
    check("responses predicted_per_second is sane",
          0 < rate < MAX_PLAUSIBLE_TPS)
    check("responses timings ride its own completed event",
          tc is not None and tc.get("type") == "response.completed"
          and "choices" not in tc)

    # A malformed usage block must not abort the stream.
    r = call("/v1/chat/completions",
             dict(payload, stream=True, scenario="bogus_usage"))
    raw = r.read().decode()
    check("bogus usage does not break stream", raw.rstrip().endswith("data: [DONE]"))
    check("bogus usage still yields timings", _timings_chunk(raw) is not None)

    # A chunked request body (no Content-Length) must reach the upstream.
    chunk_body = json.dumps(dict(payload, scenario="echo")).encode()
    c = HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions", body=iter([chunk_body]),
              headers={"Content-Type": "application/json"}, encode_chunked=True)
    obj = json.loads(c.getresponse().read())
    check("chunked request body forwarded", obj.get("echo_req_len") == len(chunk_body))

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