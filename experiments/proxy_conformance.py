"""Conformance tests for the proxy, against a fake upstream.

Runs the real proxy against a stub that records exactly what reached it, so
byte-exactness is proved rather than assumed - and proved without spending
money or depending on the network.

The checks map one-to-one onto the documented failure modes in the gateway
compatibility spec. Each assertion names the consequence of failing it.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import proxy  # noqa: E402

RECEIVED: list[dict] = []
CT_STATUS = [200]           # count_tokens status the stub returns
SSE_GAPS: list[float] = []


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a) -> None:
        pass

    def _record(self, method: str) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        RECEIVED.append({
            "method": method,
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": body,
        })
        return body

    def do_HEAD(self) -> None:
        self._record("HEAD")
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        self._record("GET")
        payload = json.dumps({"data": [{"id": "claude-opus-5-5"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        body = self._record("POST")
        if "force400" in self.path:
            payload = json.dumps({
                "type": "error",
                "error": {"type": "invalid_request_error",
                          "message": "model: nonexistent-model not found"},
            }).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("x-should-retry", "false")
            self.send_header("retry-after", "7")
            self.send_header("anthropic-ratelimit-unified-status", "allowed")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if "bind400" in self.path:
            try:
                th = json.loads(body).get("thinking") or {}
            except ValueError:
                th = {}
            if (th.get("block_binding") or {}).get("prefix_mismatch_behavior") != "drop_block":
                payload = json.dumps({"type": "error", "error": {
                    "type": "invalid_request_error",
                    "message": "messages.1.content.0: Invalid `signature` in `thinking` block. The block is "
                               "bound to a different conversation. Remove the block, or set "
                               "`thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\"."}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
        if "count_tokens" in self.path:
            if CT_STATUS[0] != 200:
                payload = json.dumps({"type": "error", "error": {"type": "x", "message": "x"}}).encode()
                self.send_response(CT_STATUS[0])
                self.send_header("retry-after", "120")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            try:
                n = len(json.dumps(json.loads(body))) // 4 if body.strip().startswith(b"{") else 1234
            except ValueError:
                n = 1234
            payload = json.dumps({"input_tokens": n if n > 20 else 1234}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # Streamed reply, with a ping and a deliberate pause so buffering shows.
        cached = 0 if "nocache" in self.path else 240_000
        events = [
            ("message_start", {"type": "message_start", "message": dict({"usage": {
                "input_tokens": 12, "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 3_000, "output_tokens": 1}},
                **({"input_transformations": [{"type": "drop_block", "path": "messages.3.content.0",
                                               "reason": "prefix_binding_mismatch"}]}
                   if "xform" in self.path else {}))}),
            ("ping", {"type": "ping"}),
            ("content_block_start", {"type": "content_block_start", "index": 0}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "hi"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {},
                               "usage": {"output_tokens": 77}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("anthropic-ratelimit-unified-status", "allowed")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for name, data in events:
            frame = f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
            self.wfile.write(f"{len(frame):X}\r\n".encode())
            self.wfile.write(frame)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            time.sleep(0.12)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def start(server_cls, handler, port=0):
    """Bind an ephemeral port by default.

    Fixed ports made this suite fail intermittently when two runs landed
    close together and the previous sockets were still in TIME_WAIT - a flaky
    conformance test is worse than none, because it teaches you to ignore it.
    """
    srv = server_cls(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def post(url: str, body: bytes, headers: dict) -> tuple[int, dict, bytes, list[float]]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    gaps, last = [], time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        status = resp.status
    except urllib.error.HTTPError as exc:
        resp, status = exc, exc.code
    chunks = []
    while True:
        part = resp.read(1)
        if not part:
            break
        if part == b"\n":
            now = time.time()
            gaps.append(now - last)
            last = now
        chunks.append(part)
    return status, {k.lower(): v for k, v in resp.headers.items()}, b"".join(chunks), gaps


def main() -> int:
    # Never touch the user's real settings or vault: the suite writes both, and
    # its stub numbers would otherwise show up in `proxy --stats`.
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    proxy.config.PATH = tmp / "config.json"
    proxy.config._cache.update(mtime=None, data={}, checked=0.0)
    proxy.vault.VAULT_DIR, proxy.vault.DB_PATH, proxy.vault.BLOB_DIR = tmp, tmp / "vault.db", tmp / "blobs"
    # The proxy read its modes from the real config when it was imported. Each
    # section turns on what it tests; everything starts off.
    proxy.REWRITE, proxy.BG_COMPACT = "", False
    up = start(ThreadingHTTPServer, Upstream)
    upstream = f"http://127.0.0.1:{up.server_address[1]}"
    proxy.UPSTREAM = upstream
    px = start(ThreadingHTTPServer, proxy.Handler)
    time.sleep(0.3)
    base = f"http://127.0.0.1:{px.server_address[1]}"
    failures = []

    def check(name: str, ok: bool, consequence: str) -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(f"{name}: {consequence}")

    # A request shaped like Claude Code's: cache_control in system, in a
    # message, and on a mid-conversation role:"system" entry.
    body_obj = {
        "model": "claude-opus-5-5",
        "max_tokens": 64,
        "stream": True,
        "system": [
            {"type": "text", "text": "ATTRIBUTION v1 fingerprint=abc"},
            {"type": "text", "text": "You are a coding assistant.",
             "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "read the file",
                 "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "python3 - <<'PY'\n" + "x=1\n" * 400 + "PY"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": ("[INFO] Downloading from central: "
                             "https://repo.maven.apache.org/artifact.jar\n") * 300,
                 "cache_control": {"type": "ephemeral"}}]},
            {"role": "system", "content": "terse mode",
             "cache_control": {"type": "ephemeral"}},
        ],
    }
    body = json.dumps(body_obj).encode()
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "extended-cache-ttl-2025-04-11,compact-2026-09-04",
        "x-api-key": "sk-ant-SECRET-should-not-be-stored",
        "authorization": "Bearer SECRET",
        "anthropic-workspace-id": "ws_123",
        "x-claude-code-session-id": "sess-conformance",
        "x-claude-code-request-class": "main",
        "x-claude-code-prompt-id": "p-1",
    }

    print("\n1. request byte-exactness and header forwarding")
    RECEIVED.clear()
    status, rheaders, payload, gaps = post(base + "/v1/messages?beta=true", body, headers)
    got = RECEIVED[-1]
    check("request body forwarded byte-for-byte", got["body"] == body,
          "a reshaped body breaks cache_control and header/body pairing")
    check("path preserved including query", got["path"] == "/v1/messages?beta=true",
          "the upstream routes on the path")
    for h in ("anthropic-version", "anthropic-beta", "anthropic-workspace-id",
              "x-api-key", "authorization", "x-claude-code-session-id",
              "x-claude-code-request-class"):
        check(f"forwarded {h}", got["headers"].get(h) == headers.get(h.lower(), headers.get(h)),
              "stripping a required header fails the request or silently disables a feature")
    check("hop-by-hop Host rewritten, not forwarded blindly",
          got["headers"].get("host") == upstream.removeprefix("http://"),
          "a stale Host breaks TLS/routing upstream")
    sent = json.loads(got["body"])
    check("system array order and attribution block position unchanged",
          sent["system"][0]["text"].startswith("ATTRIBUTION"),
          "the endpoint's positional strip fails and the block enters the cache key")
    check("cache_control preserved in system",
          sent["system"][1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"},
          "THE ~10x FAILURE: conversation bills as uncached every turn")
    check("cache_control preserved in a message",
          sent["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"},
          "THE ~10x FAILURE")
    check("cache_control preserved on a mid-conversation system message",
          sent["messages"][3]["cache_control"] == {"type": "ephemeral"},
          "THE ~10x FAILURE")

    print("\n2. streaming")
    text = payload.decode()
    check("content-type is text/event-stream",
          "text/event-stream" in rheaders.get("content-type", ""),
          "response unparseable by the client")
    check("ping event relayed", "event: ping" in text,
          "a stripped ping trips the streaming idle watchdog during a pause")
    check("message_delta relayed", "message_delta" in text,
          "Claude Code treats a missing final delta as a dropped connection")
    check("message_stop relayed", "message_stop" in text, "same")
    check("rate-limit header forwarded",
          "anthropic-ratelimit-unified-status" in rheaders,
          "usage-limit display breaks")
    slow = [g for g in gaps if g > 0.05]
    check(f"not buffered (saw {len(slow)} inter-event gaps >50ms)", len(slow) >= 4,
          "a buffering gateway stalls the UI until the response completes")

    print("\n3. error passthrough")
    status, rheaders, payload, _ = post(base + "/v1/messages?force400=1", body, headers)
    check("status 400 preserved", status == 400, "retry logic misreads the failure")
    check("error body unmodified",
          json.loads(payload)["error"]["message"] == "model: nonexistent-model not found",
          "capability-rejection recovery matches on the upstream's wording")
    check("x-should-retry forwarded", rheaders.get("x-should-retry") == "false",
          "Claude Code retries something unretryable")
    check("retry-after forwarded", rheaders.get("retry-after") == "7",
          "backoff timing breaks")

    print("\n4. other endpoints")
    req = urllib.request.Request(base + "/api/hello", method="HEAD")
    check("HEAD /api/hello answers",
          urllib.request.urlopen(req, timeout=10).status == 200,
          "the connection-warming probe fails")
    status, _, payload, _ = post(base + "/v1/messages/count_tokens",
                                 json.dumps({"model": "claude-opus-5-5"}).encode(), headers)
    check("count_tokens round-trips",
          status == 200 and json.loads(payload)["input_tokens"] == 1234,
          "/context degrades to character estimates")
    check("unknown GET path passes through",
          urllib.request.urlopen(base + "/v1/models", timeout=10).status == 200,
          "an unknown future path would break a session")

    print("\n5. observation")
    import sqlite3
    # The vault row is written after the response, on the relay thread, and
    # can contend with the retention thread: poll instead of a fixed wait
    # (a fixed 0.6 s failed under load during setup).
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            c = sqlite3.connect(proxy.vault.DB_PATH)
            n = c.execute("SELECT count(*) FROM events WHERE kind='api_request'"
                          " AND meta LIKE '%\"streaming\": true%'").fetchone()[0]
            c.close()
        except sqlite3.OperationalError:      # tables not created yet
            n = 0
        if n:
            break
        time.sleep(0.2)
    conn = sqlite3.connect(proxy.vault.DB_PATH)
    rows = conn.execute("SELECT id, meta FROM events WHERE kind='api_request'"
                        " ORDER BY id DESC LIMIT 6").fetchall()
    conn.close()
    metas = []
    for rid, mj in rows:
        m = json.loads(mj or "{}")
        if m.get("_meta_blob"):
            m = proxy.vault.read_meta(rid)
        metas.append(m)
    streamed = next((m for m in metas if m.get("streaming")), {})
    check("usage captured from the stream",
          (streamed.get("usage") or {}).get("cache_read_input_tokens") == 240_000,
          "no cache telemetry means the canary cannot work")
    check("output_tokens captured from the final message_delta",
          (streamed.get("usage") or {}).get("output_tokens") == 77,
          "output cost invisible")
    blob = json.dumps(metas)
    check("credentials NOT stored in the vault",
          "SECRET" not in blob,
          "a local file would hold the API credential")
    check("hint headers recorded", streamed.get("request_class") == "main",
          "cannot tell main turns from subagent turns")

    print("\n6. observation failure must not break relay")
    saved = proxy.vault.record
    proxy.vault.record = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vault down"))
    try:
        status, _, payload, _ = post(base + "/v1/messages", body, headers)
        check("request still succeeds with the vault broken", status == 200,
              "observation must never be able to fail a turn")
    finally:
        proxy.vault.record = saved

    print("\n7. upstream down")
    proxy.UPSTREAM = "http://127.0.0.1:9"
    status, _, payload, _ = post(base + "/v1/messages", body, headers)
    check("502 with an API-shaped error body",
          status == 502 and json.loads(payload)["error"]["type"] == "api_error",
          "an unfamiliar envelope breaks error handling")
    proxy.UPSTREAM = upstream

    print("\n8. opt-in forward rewrite (TOKENSAVER_PROXY_REWRITE=forward)")
    proxy.REWRITE = "forward"
    try:
        RECEIVED.clear()
        post(base + "/v1/messages?beta=true", body, headers)
        first = RECEIVED[-1]["body"]
        post(base + "/v1/messages?beta=true", body, headers)
        second = RECEIVED[-1]["body"]
        sent = json.loads(first)
        check("main request rewritten smaller", len(first) < len(body),
              "the opt-in does nothing")
        check("rewrite is byte-identical across requests", first == second,
              "THE ~10x FAILURE: a rewrite that varies breaks the cache every turn")
        check("system untouched by rewrite", sent["system"] == body_obj["system"],
              "the system prompt is the cached floor")
        check("cache_control preserved through rewrite",
              sent["messages"][0]["content"][0].get("cache_control") == {"type": "ephemeral"}
              and sent["messages"][2]["content"][0].get("cache_control") == {"type": "ephemeral"}
              and sent["messages"][3].get("cache_control") == {"type": "ephemeral"},
              "THE ~10x FAILURE")
        check("tool_use id and name kept",
              sent["messages"][1]["content"][0]["id"] == "t1"
              and sent["messages"][1]["content"][0]["name"] == "Bash",
              "tool_result pairing breaks and the API rejects the request")
        other = dict(headers, **{"x-claude-code-request-class": "classifier"})
        post(base + "/v1/messages", body, other)
        check("non-main request class relayed byte-for-byte", RECEIVED[-1]["body"] == body,
              "the auto-mode classifier sees vault paths and denies commands")
        bare = {k: v for k, v in headers.items() if k != "x-claude-code-request-class"}
        post(base + "/v1/messages", body, bare)
        check("missing request class relayed byte-for-byte", RECEIVED[-1]["body"] == body,
              "without hint headers the class is unknown, so nothing may change")
        saved_rw = proxy.passes.rewrite_messages
        proxy.passes.rewrite_messages = lambda m: (_ for _ in ()).throw(RuntimeError("bug"))
        try:
            status, _, _, _ = post(base + "/v1/messages", body, headers)
            check("rewrite failure sends the original bytes",
                  status == 200 and RECEIVED[-1]["body"] == body,
                  "a defect in a pass must never fail a turn")
        finally:
            proxy.passes.rewrite_messages = saved_rw
    finally:
        proxy.REWRITE = ""

    print("\n10. opt-in background compaction (TOKENSAVER_BG_COMPACT=1)")
    state_dir = Path(tempfile.mkdtemp())
    saved_state = proxy._state_path
    proxy._state_path = lambda s: state_dir / f"{s}.json"
    proxy.REWRITE, proxy.BG_COMPACT, proxy._BG_STARTED = "forward", True, True   # no worker thread
    old_part = [{"role": "user", "content": [{"type": "text", "text": f"old request {i}"}]} if i % 2 == 0
                else {"role": "assistant", "content": [{"type": "text", "text": f"old answer {i}"}]}
                for i in range(6)]
    keep = [{"role": "user", "content": [{"type": "text", "text": "the newest request"}]}]
    hist = old_part + keep
    prepared = {"cut": 6, "prefix_fp": proxy.background.fingerprint(old_part),
                "text": "[TokenSaver: background compaction.] SUMMARY OF OLD WORK",
                "old_tokens": 100, "new_tokens": 10, "gate": {"passed": True}}

    def bg_post(messages, session="sess-bg"):
        h = dict(headers, **{"x-claude-code-session-id": session})
        post(base + "/v1/messages", json.dumps(dict(body_obj, messages=messages)).encode(), h)
        return json.loads(RECEIVED[-1]["body"])["messages"]

    try:
        proxy._bg_save("sess-bg", {"pending": prepared})
        got = bg_post(hist)
        check("prepared summary NOT applied while the cache is warm",
              len(got) == len(hist), "a swap mid-session pays a full cache write and is felt")
        proxy._LIVE["sess-bg"]["last"] -= 2 * 3600
        got = bg_post(hist)
        check("applied after a long quiet spell",
              len(got) == 1 and "SUMMARY OF OLD WORK" in got[0]["content"][0]["text"]
              and got[0]["content"][1]["text"] == "the newest request",
              "the free rebuild moment is wasted")
        more = hist + [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                       {"role": "user", "content": [{"type": "text", "text": "next"}]}]
        again = bg_post(more)
        check("next request reproduces the compacted prefix byte for byte",
              json.dumps(again[:1], sort_keys=True) == json.dumps(got[:1], sort_keys=True),
              "THE ~10x FAILURE: the cache breaks on every request after the compaction")
        edited = [dict(old_part[0], content=[{"type": "text", "text": "rewound"}])] + old_part[1:] + keep
        got = bg_post(edited)
        check("history no longer matching: summary dropped, original sent",
              len(got) == len(edited) and not proxy._bg_load("sess-bg").get("applied"),
              "a summary of a different history would be wrong context")
        proxy._bg_save("sess-bg2", {"pending": dict(prepared, gate={"passed": False})})
        bg_post(hist, "sess-bg2")
        proxy._LIVE["sess-bg2"]["last"] -= 2 * 3600
        got = bg_post(hist, "sess-bg2")
        check("a summary that failed the QA gate is never applied", len(got) == len(hist),
              "a low-quality summary loses context")
        # No local model at all (most machines): the compaction is rules only.
        big = []
        for i in range(80):
            big += [{"role": "user", "content": [{"type": "text", "text": f"step {i}"}]},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": f"r{i}", "name": "Bash",
                                                       "input": {"command": f"run {i}"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"r{i}",
                                                  "content": "\n".join(f"log {i}.{k} ok" for k in range(300))}]},
                    {"role": "assistant", "content": [{"type": "text", "text": f"step {i} done"}]}]
        big += keep
        saved_ol = proxy.background.ollama
        proxy.background.ollama = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("no ollama here"))
        try:
            made = proxy.background.prepare(big)
        finally:
            proxy.background.ollama = saved_ol
        proxy._bg_save("sess-bg5", {"pending": made})
        bg_post(big, "sess-bg5")
        proxy._LIVE["sess-bg5"]["last"] -= 2 * 3600
        got = bg_post(big, "sess-bg5")
        check("no local model installed: compaction still prepared and applied after a break",
              bool(made) and made["gate"]["passed"] and len(got) < len(big)
              and got[0]["content"][0]["text"].startswith("[TokenSaver: background compaction"),
              f"machines without the 35B lose most of the saving "
              f"({made and (made['old_tokens'], made['new_tokens'])}, {len(got)}/{len(big)}, "
              f"{ {k: v for k, v in proxy._bg_load('sess-bg5').items() if k not in ('pending', 'applied')} })")
        got = bg_post(hist, "sess-bg3")
        check("a session never seen before never compacts",
              len(got) == len(hist), "an unknown gap may be a warm cache")
        proxy._bg_save("sess-bg4", {"pending": prepared, "last": time.time() - 2 * 3600})
        proxy._LIVE.pop("sess-bg4", None)          # as after a proxy restart
        got = bg_post(hist, "sess-bg4")
        check("after a proxy restart, the saved idle time still allows the compaction",
              len(got) == 1 and "SUMMARY OF OLD WORK" in got[0]["content"][0]["text"],
              "every restart would forfeit the saving")
        # Warm cache (idle past bg_after_min but under the 1h TTL): only if it repays.
        for sess, n, want in (("sess-bg5", 182.0, False), ("sess-bg6", 40.0, True)):
            proxy._bg_save(sess, {"pending": dict(prepared, payback_requests=n),
                                  "last": time.time() - 20 * 60})
            proxy._LIVE.pop(sess, None)
            got = bg_post(hist, sess)
            check(f"warm cache, payback {n:g} requests: {'applied' if want else 'not applied'}",
                  (len(got) == 1) == want,
                  "a warm compaction that cannot repay costs more than it saves (measured: $2.05)"
                  if not want else "a warm compaction that repays quickly is a saving thrown away")
        proxy._bg_save("sess-bg7", {"pending": dict(prepared), "last": time.time() - 20 * 60})
        proxy._LIVE.pop("sess-bg7", None)
        check("warm cache with no payback estimate: not applied",
              len(bg_post(hist, "sess-bg7")) == len(hist), "a guess instead of a measurement")
        live = {"sent_tokens": 384_000, "skeleton": {"model": "claude-opus-5-5"}}
        # Trial numbers: 384K sent now, 317K after (the estimate scales what is
        # sent now by the compacted/current size ratio: 284K of 1,627.5K removed).
        pb = proxy._payback({"old_tokens": 440_000, "new_tokens": 156_000, "total_tokens": 1_627_500}, {}, live)
        check("payback measured against what is sent now (trial: 384K -> 317K needs ~180 requests)",
              170 < (pb.get("payback_requests") or 0) < 195, f"got {pb}")
        pb = proxy._payback({"old_tokens": 1, "new_tokens": 1, "total_tokens": 390_000}, {}, live)
        check("a compaction that sends no fewer tokens never applies warm",
              pb.get("payback_requests") is None, "re-applying the same summary paid a full write")
        applied = {"cut": 481, "total_tokens": 500_000}
        check("no new compaction at the cut already in force",
              not proxy._new_cut_ok({"cut": 481}, applied) and proxy._new_cut_ok({"cut": 585}, applied),
              "the same summary would be applied again")
        small = [{"role": "user", "content": [{"type": "text", "text": "x" * 100}]}]
        check("no re-prepare right after an apply (pending empty, applied set)",
              not proxy._prepare_due({"pending": None, "applied": {"total_tokens": 10**9}}, small)
              and proxy._prepare_due({}, small),
              "the worker re-prepared at once after every apply")
    finally:
        proxy.REWRITE, proxy.BG_COMPACT = "", False
        proxy._state_path = saved_state

    print("\n11. automatic fallback when the cache stops being read")
    proxy.REWRITE = "forward"
    try:
        def fb_post(session, nocache=True):
            h = dict(headers, **{"x-claude-code-session-id": session})
            post(base + "/v1/messages" + ("?nocache=1" if nocache else ""), body, h)
            time.sleep(0.2)               # _watch runs after the relay
            return RECEIVED[-1]["body"]
        for _ in range(4):                # first is cold (ok), then 3 warm misses
            fb_post("sess-fb1")
        check("session falls back to plain relay after 3 warm cache misses",
              fb_post("sess-fb1") == body and "sess-fb1" in proxy._FELL_BACK,
              "a rewrite that breaks the cache would bill every turn uncached")
        check("other sessions keep rewriting after one fallback",
              fb_post("sess-fb-ok", nocache=False) != body,
              "one bad session should not stop the saving everywhere")
        for _ in range(4):
            fb_post("sess-fb2")
        check("two fallbacks in a day switch rewriting off globally",
              bool(proxy._global_off()) and fb_post("sess-fb3", nocache=False) == body,
              "repeated cache breaks must stop the proxy rewriting at all")
        proxy.config.set_section("fallback", {})
        proxy._FELL_BACK.clear()
        check("clearing the fallback resumes rewriting",
              fb_post("sess-fb4", nocache=False) != body, "the off switch must be reversible")
    finally:
        proxy.REWRITE = ""

    print("\n12. token audit (count_tokens, after the relay)")
    from tokensaver import audit
    proxy.REWRITE = "forward"
    audit.UPSTREAM, audit.MIN_GAP_S, audit.SETTLE_S = upstream, 0, 0
    audit._per_session.clear()
    try:
        def ct_calls():
            return [r for r in RECEIVED if "count_tokens" in r["path"]]
        RECEIVED.clear()
        for i in range(3):
            post(base + "/v1/messages", body, dict(headers, **{"x-claude-code-session-id": "sess-a0"}))
        time.sleep(0.5)
        check("audit off: no count_tokens calls", not ct_calls(),
              "no extra API calls unless audit is on")
        proxy.config.set_value("audit", "on", "user")
        RECEIVED.clear()
        post(base + "/v1/messages", body, dict(headers, **{"x-claude-code-session-id": "sess-a1"}))
        main_body = RECEIVED[0]["body"]
        time.sleep(1.0)
        calls = ct_calls()
        # The pair, then one count per pass that changed something (toggled).
        check("audit on: the count_tokens pair (+ one per pass that changed) after the main response",
              2 <= len(calls) <= 2 + len(audit.ATTRIBUTE) and RECEIVED.index(calls[0]) > 0,
              "the audit must never run before or inside the real request")
        sent_keys = set(json.loads(calls[0]["body"])) if calls else set()
        check("count body is allowlisted (no stream/max_tokens/metadata)",
              bool(sent_keys) and sent_keys <= set(audit.BODY_FIELDS),
              "count_tokens rejects fields it does not take")
        h = calls[0]["headers"] if calls else {}
        check("count call carries auth, never x-claude-code-* headers",
              ("x-api-key" in h or "authorization" in h) and not any(k.startswith("x-claude-code") for k in h),
              "an audit call must not look like a turn")
        RECEIVED.clear()
        post(base + "/v1/messages", body, dict(headers, **{"x-claude-code-session-id": "sess-a2"}))
        check("main request body identical with audit on", RECEIVED[0]["body"] == main_body,
              "the audit must not change what is sent")
        time.sleep(1.0)
        import sqlite3
        conn = sqlite3.connect(proxy.vault.DB_PATH)
        row = conn.execute("SELECT meta FROM events WHERE kind='token_audit' ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        meta = json.loads(row[0]) if row else {}
        check("token_audit row recorded with both counts and no secrets",
              meta.get("tokens_original", 0) > meta.get("tokens_sent", 0) > 0 and "SECRET" not in json.dumps(meta),
              "no real token numbers otherwise")
        check("counts are thinking-fair: both bodies counted without thinking",
              meta.get("method") == "no_thinking" and meta.get("tokens_original_nt", 0) > meta.get("tokens_sent_nt", 0),
              "the original's dropped thinking made plain Claude look cheaper (2026-10-22)")
        check("every forward pass priced on the request (saving, on/off, changed)",
              set(meta.get("passes") or {}) >= set(audit.ATTRIBUTE) - {"path_aliases"}
              and all({"on", "saving", "changed"} <= set(v) for v in (meta.get("passes") or {}).values()),
              "a pass that costs tokens could never be found")
        CT_STATUS[0] = 429
        RECEIVED.clear()
        post(base + "/v1/messages", body, dict(headers, **{"x-claude-code-session-id": "sess-a3"}))
        time.sleep(1.0)
        n429 = len(ct_calls())
        post(base + "/v1/messages", body, dict(headers, **{"x-claude-code-session-id": "sess-a4"}))
        time.sleep(1.0)
        check("429 pauses the audit (no further calls in the window)",
              n429 == 1 and len(ct_calls()) == 1 and audit._state["pause_until"] > time.time(),
              "hammering a rate-limited endpoint")
        audit._state.update(pause_until=0.0, backoff=0.0)
        CT_STATUS[0] = 401
        post(base + "/v1/messages", body, dict(headers, **{"x-claude-code-session-id": "sess-a5"}))
        time.sleep(1.0)
        check("401 turns the audit off for the process", bool(audit._state["disabled"]),
              "retrying with credentials the endpoint refuses")
    finally:
        CT_STATUS[0] = 200
        audit._state.update(disabled=None, pause_until=0.0, backoff=0.0)
        proxy.config.unset("audit", "user")
        proxy.REWRITE = ""

    print("\n13. aliases decided at free points, frozen between them")
    proxy.REWRITE = "forward"
    try:
        P = "/Users/someone/Projects/Java/a-long-service-name/src/main/java/com/example"
        paths = [{"role": "user", "content": [{"type": "text", "text": f"look at {P}/F{i}.java and {P}/G{i}.java"}]}
                 if i % 2 == 0 else {"role": "assistant", "content": [{"type": "text", "text": f"ok {P}/H{i}.java"}]}
                 for i in range(7)]

        def al_post(session):
            h = dict(headers, **{"x-claude-code-session-id": session})
            post(base + "/v1/messages", json.dumps(dict(body_obj, messages=paths)).encode(), h)
            return RECEIVED[-1]["body"].decode()
        check("a new session gets no aliases",
              "TokenSaver path aliases" not in al_post("sess-al-new"),
              "defined-as-they-go aliases measured -2.4 to -3.3% tokens")
        proxy._bg_save("sess-al-auto", {"aliases": {"mode": "auto"}})
        check("a stored 'auto' mode (removed) is treated as off",
              "TokenSaver path aliases" not in al_post("sess-al-auto"), "auto aliases cost tokens")
        plan = proxy.passes.hindsight_plan(paths)
        proxy._bg_save("sess-al-plan", {"aliases": {"mode": "plan", "plan": plan}})
        a, b = al_post("sess-al-plan"), al_post("sess-al-plan")
        check("a frozen plan is applied and reproduces the same bytes",
              bool(plan) and a == b and "TokenSaver path aliases" in a,
              "THE ~10x FAILURE: a plan that varied would break the cache every turn")
        rewound = paths[:5]
        h = dict(headers, **{"x-claude-code-session-id": "sess-al-plan"})
        post(base + "/v1/messages", json.dumps(dict(body_obj, messages=rewound)).encode(), h)
        check("a rewind (same start, fewer messages) keeps the plan",
              "TokenSaver path aliases" in RECEIVED[-1]["body"].decode(),
              "the rewound prefix is still cached with the plan's bytes")
        compacted = [{"role": "user", "content": [{"type": "text", "text": "This session is being continued "
                                                  f"from a previous conversation. {P}/F0.java"}]}] + paths[1:]
        post(base + "/v1/messages", json.dumps(dict(body_obj, messages=compacted)).encode(), h)
        check("Claude Code's /compact (a new history start) turns the plan off",
              "TokenSaver path aliases" not in RECEIVED[-1]["body"].decode()
              and proxy._bg_load("sess-al-plan")["aliases"]["since"] == "history rebuilt",
              "measured: the old plan cost 4.7% tokens after a /compact")
    finally:
        proxy.REWRITE = ""

    print("\n15. forward passes measured as a net cost go off, at free points only")
    proxy.REWRITE = "forward"
    saved_verdict = dict(proxy._VERDICT)
    try:
        def put(sess, n, passes):
            for i in range(n):
                proxy.vault.record("token_audit", session_id=sess, summary="x", index=False,
                                   meta={"tokens_sent": 100_000, "passes": passes, "method": "no_thinking"})
        put("sess-pv", proxy.PASS_MIN_EVIDENCE,
            {"format": {"on": True, "saving": -500, "changed": True, "chars": 2_000},
             "json_compact": {"on": True, "saving": -6_000, "changed": True, "chars": 13}})
        proxy._VERDICT["at"] = 0.0
        off = proxy.pass_verdicts()
        check("evidence from one session alone switches nothing off", off == (), f"verdict {off}")
        put("sess-pv2", proxy.PASS_PER_SESSION, {"format": {"on": True, "saving": -500, "changed": True, "chars": 2_000},
                            "repeat_runs": {"on": True, "saving": 800, "changed": True},
                            "json_compact": {"on": True, "saving": -6_000, "changed": True, "chars": 13}})
        put("sess-pv", proxy.PASS_MIN_EVIDENCE, {"repeat_runs": {"on": True, "saving": 800, "changed": True}})
        proxy._VERDICT["at"] = 0.0
        off = proxy.pass_verdicts()
        check("a pass that measured as a net cost is switched off; one that saves stays on",
              "format" in off and "repeat_runs" not in off, f"verdict {off}")
        check("a count larger than the characters toggled is ignored", "json_compact" not in off,
              f"verdict {off}")
        proxy._bg_save("sess-pv-old", {"passes_off": [], "hist_fp": None})
        h = dict(headers, **{"x-claude-code-session-id": "sess-pv-old"})
        a1 = proxy._session_setup("sess-pv-old", body_obj["messages"], 30)
        check("a session mid-cache keeps its passes (no free point)", a1[1] == (),
              "changing a cached session's bytes forces a full rebuild")
        a2 = proxy._session_setup("sess-pv-old", body_obj["messages"], 2 * 3600)
        check("an expired cache adopts the verdict", "format" in a2[1], "the free rebuild is wasted")
        a3 = proxy._session_setup("sess-pv-new", body_obj["messages"], None)
        check("a new session starts with the verdict", "format" in a3[1], "new sessions would pay the cost")
    finally:
        proxy._VERDICT.update(saved_verdict, at=0.0)
        proxy.config.set_section("passes", {})
        proxy.REWRITE = ""

    print("\n16. measuring is an enhancement: with counts unavailable, safe defaults")
    proxy.REWRITE = "forward"
    saved_count, saved_enabled = proxy.audit._count, proxy.audit.enabled
    saved_verdict = dict(proxy._VERDICT)
    try:
        proxy.audit.enabled = lambda: True
        proxy.audit._count = lambda path, h, body: (None, 0, {})          # endpoint down
        P2 = "/Users/someone/Projects/x/a-long-service-name/src/main/java"
        hist = [{"role": "user", "content": [{"type": "text", "text": f"{P2}/A{i}.java {P2}/B{i}.java"}]}
                for i in range(5)]
        live = {"headers": {"x-api-key": "k"}, "skeleton": {"model": "claude-opus-5-5"}, "sent_tokens": 300_000}
        proxy.audit.enabled = lambda: False
        pb = proxy._payback({"old_tokens": 400_000, "new_tokens": 150_000, "total_tokens": 500_000,
                             "alias_choice": {"mode": "off", "why": "count failed (0)"}}, {}, live)
        check("no counts: payback still estimated locally (never a crash)",
              isinstance(pb.get("payback_requests"), float), f"got {pb}")
        check("no live size: warm apply refused, cold apply still allowed",
              proxy._payback({"old_tokens": 1, "new_tokens": 0}, {}, {})["payback_requests"] is None
              and not proxy._warm_ok({"payback_requests": None})[0], "a guess would pay a full write")
        # old evidence (more than PASS_STALE_S ago) is not evidence
        proxy.config.set_section("passes", {"off": ["format"]})
        conn = __import__("sqlite3").connect(proxy.vault.DB_PATH)
        conn.execute("UPDATE events SET ts='2000-01-01T00:00:00Z' WHERE kind='token_audit'")
        conn.commit(); conn.close()
        proxy._VERDICT["at"] = 0.0
        check("no recent counts: every pass back on (the validated defaults)",
              proxy.pass_verdicts() == () and not proxy.config.section("passes").get("off"),
              "a pass switched off once would stay off forever")
        saved_db = proxy.vault.DB_PATH
        proxy.vault.DB_PATH = Path(tempfile.mkdtemp()) / "missing" / "nope.db"
        proxy._VERDICT["at"] = 0.0
        try:
            check("vault unreadable: every pass on, no exception",
                  proxy.pass_verdicts() == () and proxy._session_setup("s16b", hist, None)[1] == (),
                  "a broken vault must not break rewriting")
        finally:
            proxy.vault.DB_PATH = saved_db
        from tokensaver import sim as _sim
        check("token factor with no measured compactions: the default",
              _sim.token_factor()[0] == _sim.TOKENS_PER_ESTIMATED_SAVED, "the sim would use a guess")
    finally:
        proxy.audit._count, proxy.audit.enabled = saved_count, saved_enabled
        proxy._VERDICT.update(saved_verdict, at=0.0)
        proxy.config.set_section("passes", {})
        proxy.REWRITE = ""

    print("\n17. cache markers go out exactly as Claude Code set them")
    # cache_tail_5m (2026-10-16 to 2026-10-22) moved the newest marker to 5m.
    # Measured on a real session: the next request read that piece 2 times in 66 and
    # wrote it again at 1h the other 64, so every 5m write was extra cost.
    # Removed; this guards against it coming back by accident.
    proxy.REWRITE = "forward"
    try:
        b17 = dict(body_obj, messages=[
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a",
                                               "cache_control": {"type": "ephemeral", "ttl": "1h"}}]},
            {"role": "user", "content": [{"type": "text", "text": "new",
                                          "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}])
        post(base + "/v1/messages", json.dumps(b17).encode(), dict(headers, **{"x-claude-code-session-id": "sess-ttl"}))
        got = json.loads(RECEIVED[-1]["body"])["messages"]
        check("every cache marker keeps its lifetime",
              [x.get("cache_control") for m in got for x in m["content"] if x.get("cache_control")]
              == [{"type": "ephemeral", "ttl": "1h"}] * 2,
              "a 5m newest piece is written twice: 5m now, 1h on the next request")
    finally:
        proxy.REWRITE = ""

    print("\n18. staged pruning: a candidate, one final check, rules-only until then")
    saved = (proxy.prune.tier, proxy.prune.run_stage, proxy.background.gate, proxy.prune.unload,
             proxy.prune.line_picker, proxy.prune.can_load, proxy._check_model)
    state_dir18 = Path(tempfile.mkdtemp())
    saved_bgpath = proxy._bg_path
    proxy._bg_path = lambda s: state_dir18 / f"{s}.bg.json"
    try:
        log18 = "\n".join(f"step {i}: compiled module {i} without warnings" for i in range(80))
        old18 = [{"role": "user", "content": [{"type": "text", "text": "build"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "an old answer " * 40},
                                                   {"type": "tool_use", "id": "b1", "name": "Bash",
                                                    "input": {"command": "make"}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b1", "content": log18}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "done"}]}]
        hist18 = old18 + [{"role": "user", "content": [{"type": "text", "text": "the newest request"}]}]
        pend = {"cut": 4, "prefix_fp": proxy.background.fingerprint(old18), "text": "RULES-ONLY",
                "old_tokens": 500, "new_tokens": 10 ** 6, "gate": dict(proxy.background.RULES_ONLY),
                "stage": "extract", "made": 1.0}
        live = {"messages": hist18, "last": time.time() - 3600, "skeleton": {}, "headers": {}}
        proxy.prune.unload = lambda m: None
        proxy.prune.can_load = lambda m: (True, "ok")
        proxy.prune.run_stage = lambda *a, **k: ({id(old18[1]["content"][0])}, {"deleted": 1})
        proxy.prune.line_picker = lambda *a, **k: ((lambda kind, text: text.split("\n")[0]), {"picked": 1})
        proxy._check_model = lambda: "the-35b"
        gates: list = []

        def run(name, steps=6):
            proxy._bg_save(name, {"pending": dict(pend)})
            seen = []
            for _ in range(steps):
                proxy._prune_step(name, live)
                seen.append(proxy._bg_load(name)["pending"])
                if seen[-1].get("prune_done"):
                    break
            return seen

        proxy.prune.tier = lambda: ("deep", "test")
        proxy.background.gate = lambda old, cand, **kw: (gates.append(kw), {"passed": True})[1]
        seen = run("s18")
        check("stages never replace the rules-only compaction on the way",
              all(x["text"] == "RULES-ONLY" for x in seen[:-1]) and len(seen) == 3,
              "a half-done, unchecked prune could be applied at a break")
        got = seen[-1]
        check("after the last stage one check (standard + current work) against rules-only promotes it",
              got["prune_done"] and "removed here" in got["text"] and "an old answer" not in got["text"]
              and len(gates) == 2 and "current" in gates[1]
              and all(kw.get("original_for") for kw in gates) and got["gate"].get("check", "").startswith("final"),
              f"checks {len(gates)}, stage {got.get('stage')}")
        proxy.background.gate = lambda old, cand, **kw: {"passed": "current" not in kw}
        got = run("s18b")[-1]
        check("a failed final check keeps rules-only", got["prune_done"] and got["text"] == "RULES-ONLY",
              "a lossy prune would be applied")
        proxy.prune.tier = lambda: ("fast", "test")
        proxy._check_model = lambda: None
        got = run("s18c")[-1]
        check("fast tier without the 35B: promoted unchecked after the fast stage",
              got["prune_done"] and got["stage"] == "fast" and "removed here" in got["text"]
              and "no check model" in got["gate"].get("check", ""), "Nimble-only machines would gain nothing")
        proxy.prune.run_stage = lambda *a, **k: (None, {"stopped": "machine strained (stop)"})
        got = run("s18d", 2)[-1]
        check("a stage stopped by strain changes nothing", got["text"] == "RULES-ONLY" and not got.get("prune_done")
              and not got.get("pruned"), "a half-done stage would be stored")
        proxy.prune.tier = lambda: ("deep", "test")
        proxy.prune.run_stage = lambda *a, **k: ({id(old18[1]["content"][0])}, {"deleted": 1})
        proxy.prune.line_picker = lambda *a, **k: ((lambda kind, text: text), {"stopped": "machine strained (stop)"})
        proxy._check_model = lambda: "the-35b"
        got = run("s18e", 4)[-1]
        check("a strained machine pauses the lines stage", got["text"] == "RULES-ONLY" and not got.get("prune_done")
              and (got.get("pruned") or {}).get("tried") == "deep", "half-picked text would be applied")
        proxy.prune.tier = lambda: ("off", "test")
        proxy._bg_save("s18f", {"pending": dict(pend)})
        check("prune off: nothing runs", not proxy._prune_step("s18f", live), "the switch must work")
    finally:
        (proxy.prune.tier, proxy.prune.run_stage, proxy.background.gate, proxy.prune.unload,
         proxy.prune.line_picker, proxy.prune.can_load, proxy._check_model) = saved
        proxy._bg_path = saved_bgpath

    print("\n19. sent-bytes ledger: an unchanged prefix goes out as the bytes sent before")
    proxy.REWRITE = "forward"
    saved_rw, saved_sent = proxy.passes.rewrite_messages, dict(proxy._SENT)
    import random as _random
    def noisy(messages, **kw):      # a nondeterministic pass: different bytes on every call
        out = saved_rw(messages, **kw)
        m0 = dict(out[0]); c = [dict(x) for x in m0["content"]]
        c[0]["text"] = c[0]["text"] + f" [{_random.random()}]"
        m0["content"] = c
        return [m0] + out[1:]
    try:
        proxy.passes.rewrite_messages = noisy
        h19 = dict(headers, **{"x-claude-code-session-id": "s19"})
        for _s in ("s19", "s19-off"):     # sessions the proxy has seen (else: adopted raw, section 22)
            proxy._bg_save(_s, {})
        a19 = [dict(m) for m in body_obj["messages"][:3]]
        b19 = a19 + [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                     {"role": "user", "content": [{"type": "text", "text": "next",
                                                   "cache_control": {"type": "ephemeral"}}]}]
        def sent(msgs, sess="s19"):
            out, st = proxy._rewrite(json.dumps(dict(body_obj, messages=msgs)).encode(), "/v1/messages",
                                     "main", sess, dict(headers, **{"x-claude-code-session-id": sess}))
            return json.loads(out)["messages"], st
        canon = lambda ms: json.dumps(proxy._canon(ms), sort_keys=True)
        s1, _ = sent(a19)
        s2, st2 = sent(b19)
        check("a nondeterministic rewrite still sends the stored bytes for the prefix",
              canon(s2[:3]) == canon(s1[:3]) and st2.get("drift") == 1 and st2.get("replayed") == 3,
              "every recomputation that differs breaks the cache")
        proxy._SENT.clear()
        s3, st3 = sent(b19 + [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}])
        check("after a restart the ledger on disk replays the same bytes", canon(s3[:5]) == canon(s2[:5]),
              "a restarted proxy would rebuild the cache")
        n1, _ = sent(a19, "s19-off")
        proxy._ledger_path("s19-off").unlink()
        proxy._SENT.pop("s19-off", None)
        n2, _ = sent(b19, "s19-off")
        check("negative control: without the ledger the noisy pass changes the prefix",
              canon(n2[:3]) != canon(n1[:3]), "the fault injection proves nothing")
        moved = [dict(m) for m in b19]
        moved[4] = {"role": "user", "content": [{"type": "text", "text": "next"}]}
        moved += [{"role": "assistant", "content": [{"type": "text", "text": "yes"}]}]
        s4, _ = sent(moved)
        check("a replayed message takes this request's cache_control placement",
              "cache_control" not in s4[4]["content"][0] and canon(s4[:5]) == canon(s2[:5]),
              "stale cache markers would be sent")
        e1 = proxy.estimate("s19")
        n = proxy._NOW["s19"]
        check("the status line estimate is what the last request saved; a small session is done",
              e1.get("pct") == round(100 * (1 - n["out"] / n["in"])) and e1.get("done") is True,
              f"estimate {e1}")
        st19 = proxy._bg_load("s19")
        st19["pending"] = {"gate": {"passed": True}, "old_tokens": 50_000, "new_tokens": 10_000, "cut": 2,
                           "payback_requests": 5}
        proxy._bg_save("s19", st19)
        saved_bgc = proxy.BG_COMPACT
        proxy.BG_COMPACT = True
        n["t"] -= proxy.BG_APPLY_AFTER_S + 60
        e2 = proxy.estimate("s19")
        n["t"] = time.time()
        e3 = proxy.estimate("s19")
        proxy.BG_COMPACT = saved_bgc
        check("a ready compaction raises the estimate once the pause is long enough, not before",
              e2["pct"] > e1["pct"] and e3["pct"] == e1["pct"] and e3.get("next_pct") == e2["pct"]
              and e3["done"] is False,
              f"{e1} {e2} {e3}")
        proxy.audit.LATEST["s19"] = {"orig": 200_000, "sent": 150_000, "bytes_in": 1, "bytes_out": 1, "t": 0}
        e4 = proxy.estimate("s19")
        proxy.audit.LATEST.pop("s19")
        check("with a real count the badge shows real tokens saved, not bytes",
              e4.get("basis") == "tokens" and e4.get("pct") == 25 and e1.get("basis") == "bytes",
              f"{e4} {e1}")
        st19.pop("pending"); proxy._bg_save("s19", st19)
        _, st5 = proxy._replay("s19", proxy._SENT["s19"]["sent"][:1], [{"role": "user", "content": "x"}],
                               "another-epoch", update=False)
        check("a new epoch (free point) replays nothing", "replayed" not in st5,
              "a compaction or a new alias plan could never take effect")
    finally:
        proxy.passes.rewrite_messages = saved_rw
        proxy._SENT.clear(); proxy._SENT.update(saved_sent)
        proxy.REWRITE = ""

    print("\n20. preserved thinking: bytes Claude's thinking is bound to never change")
    proxy.REWRITE = "forward"
    try:
        from tokensaver import background as bg20, passes as ps20
        think = lambda: {"type": "thinking", "thinking": "", "signature": "sig"}
        h20 = dict(headers, **{"x-claude-code-session-id": "s20"})
        b20 = dict(body_obj, thinking={"type": "adaptive"}, messages=body_obj["messages"][:3])
        n0 = len(RECEIVED)
        st, _, out, _ = post(base + "/v1/messages?bind400", json.dumps(b20).encode(), h20)
        tries = RECEIVED[n0:]
        retry = json.loads(tries[-1]["body"]) if tries else {}
        check("a thinking-binding 400 is retried once with drop_block, and the client gets the reply",
              st == 200 and len(tries) == 2 and (retry.get("thinking") or {}).get("block_binding")
              == {"prefix_mismatch_behavior": "drop_block"}
              and proxy.BINDING_BETA in tries[-1]["headers"].get("anthropic-beta", ""),
              "on accounts from 2026-08-31 every such request would fail in front of the user")
        post(base + "/v1/messages", json.dumps(b20).encode(), h20)
        check("the session keeps drop_block from then on (stable bytes)",
              (json.loads(RECEIVED[-1]["body"]).get("thinking") or {}).get("block_binding")
              == {"prefix_mismatch_behavior": "drop_block"} and proxy._bg_load("s20").get("drop_block"),
              "every later request would 400 and retry")
        raw400 = json.dumps(dict(b20, model="nonexistent-model")).encode()
        st, _, out400, _ = post(base + "/v1/messages?force400", raw400,
                                dict(headers, **{"x-claude-code-session-id": "s20-other"}))
        check("any other 400 is relayed as it came, with no retry",
              st == 400 and b"nonexistent-model not found" in out400
              and json.loads(RECEIVED[-1]["body"]).get("thinking", {}).get("block_binding") is None,
              "Claude Code's error recovery matches on the upstream's wording")
        post(base + "/v1/messages", json.dumps(b20).encode(),
             dict(headers, **{"x-claude-code-session-id": "s20-mon"}))
        check("rewritten requests ask the API to report failing thinking blocks (header only)",
              proxy.BINDING_BETA in RECEIVED[-1]["headers"].get("anthropic-beta", "")
              and "block_binding" not in json.loads(RECEIVED[-1]["body"]).get("thinking", {}),
              "a broken binding would stay invisible")
        post(base + "/v1/messages?xform", json.dumps(b20).encode(),
             dict(headers, **{"x-claude-code-session-id": "s20-x"}))
        time.sleep(0.2)
        import sqlite3 as _sq
        row = _sq.connect(proxy.vault.DB_PATH).execute(
            "SELECT id FROM events WHERE kind='api_request' AND session_id='s20-x' ORDER BY id DESC").fetchone()
        check("dropped blocks reported by the API are recorded",
              row and proxy.vault.read_meta(row[0]).get("thinking_dropped") == 1,
              "the stats line could never show a binding failure")
        # A pass verdict changing at a free point must not re-render sent bytes.
        tr = "log   \n\x1b[31mred\x1b[0m\nend  "
        h1 = [{"role": "user", "content": [{"type": "text", "text": "go"}]},
              {"role": "assistant", "content": [think(), {"type": "tool_use", "id": "t1", "name": "Bash",
                                                         "input": {"command": "x"}}]},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tr}]}]
        def sent20(msgs):
            o, _ = proxy._rewrite(json.dumps(dict(body_obj, messages=msgs)).encode(), "/v1/messages",
                                  "main", "s20-v", dict(headers, **{"x-claude-code-session-id": "s20-v"}))
            return json.loads(o)["messages"]
        first = sent20(h1)
        stv = proxy._bg_load("s20-v"); stv["passes_off"] = ["format"]; proxy._bg_save("s20-v", stv)
        h2 = h1 + [{"role": "assistant", "content": [think(), {"type": "tool_use", "id": "t2", "name": "Bash",
                                                              "input": {"command": "y"}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": tr}]}]
        second = sent20(h2)
        canon = lambda ms: json.dumps(proxy._canon(ms), sort_keys=True)
        check("a pass switched off mid-session changes only blocks sent after it",
              first[2] != h1[2] and canon(second[:3]) == canon(first[:3])
              and second[4]["content"][0]["content"] == tr,
              "re-rendering sent bytes invalidates every later thinking block (400 on new accounts)")
        # The kept tail of a compaction carries no thinking from before it.
        msgs = [{"role": "user", "content": [{"type": "text", "text": "old " * 50}]},
                {"role": "assistant", "content": [think(), {"type": "text", "text": "a"}]},
                {"role": "user", "content": [{"type": "text", "text": "keep"}]},
                {"role": "assistant", "content": [think(), {"type": "text", "text": "b"}]},
                {"role": "user", "content": [{"type": "text", "text": "more"}]}]
        prep = {"cut": 2, "prefix_fp": bg20.fingerprint(msgs[:2]), "text": "SUMMARY", "tail_end": 5}
        later = msgs + [{"role": "assistant", "content": [think(), {"type": "text", "text": "c"}]},
                        {"role": "user", "content": [{"type": "text", "text": "next"}]}]
        app = bg20.apply(later, prep)
        kinds = [[x.get("type") for x in m["content"]] for m in app]
        check("compaction strips thinking from the kept tail, and keeps thinking made after it",
              "thinking" not in kinds[1] and kinds[3][0] == "thinking",
              "the tail's thinking is bound to the history the summary replaced")
        dup = "a\nb\nb\nb\nb\nc"
        um = [{"role": "user", "content": [{"type": "text", "text": dup}]}]
        check("the user's own words are never edited by the passes",
              ps20.rewrite_messages(um, aliases="off")[0]["content"][0]["text"] == dup,
              "dedup removed lines from a page the user pasted (2026-10-20)")
    finally:
        proxy.REWRITE = ""

    print("\n21. the free-point decision uses real counts")
    proxy.REWRITE, proxy.BG_COMPACT, proxy._BG_STARTED = "forward", True, True
    saved_count, saved_enabled = proxy.audit._count, proxy.audit.enabled
    try:
        from tokensaver import background as bg21
        proxy.audit.enabled = lambda: True
        old21 = [{"role": "user", "content": [{"type": "text", "text": f"old request {i} " * 40}]} if i % 2 == 0
                 else {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "s" * 50},
                                                        {"type": "text", "text": f"old answer {i} " * 40}]}
                 for i in range(6)]
        hist21 = old21 + [{"role": "user", "content": [{"type": "text", "text": "the newest request"}]}]
        prep21 = {"cut": 6, "prefix_fp": bg21.fingerprint(old21), "text": "SUMMARY", "made": 1.0,
                  "old_tokens": 1000, "new_tokens": 10, "gate": {"passed": True}}
        def run21(sess, count, gap_s, prep=prep21):
            proxy._bg_save(sess, {"pending": dict(prep), "last": time.time() - gap_s})
            proxy._LIVE.pop(sess, None)
            proxy.audit._count = count
            out, st = proxy._rewrite(json.dumps(dict(body_obj, thinking={"type": "adaptive"}, messages=hist21)).encode(),
                                     "/v1/messages", "main", sess,
                                     dict(headers, **{"x-claude-code-session-id": sess, "x-api-key": "k"}))
            return json.loads(out)["messages"], st
        by_len = lambda path, h, body: (len(json.dumps(body)) // 4, 200, {})
        got, st = run21("s21a", by_len, 2 * 3600)
        check("cold, counted smaller: applied, with the counts recorded",
              len(got) == 1 and st.get("bg_applied") is True and "now" in (st.get("bg_counts") or {}),
              f"got {len(got)} messages, {st.get('bg_why')}")
        big = dict(prep21, text="SUMMARY " * 5000)
        got, st = run21("s21b", by_len, 2 * 3600, big)
        check("cold, counted BIGGER: not applied, though the estimate said smaller",
              len(got) == len(hist21) and "counted" in str(st.get("bg_skipped")),
              "the estimate once called a compaction smaller that billed ~23K tokens more")
        got, st = run21("s21c", lambda path, h, body: (None, 0, {}), 2 * 3600)
        check("counts unavailable: falls back to the estimate (cold, clear saving: applied)",
              len(got) == 1, f"{st.get('bg_why')}")
        got, st = run21("s21d", lambda path, h, body: (None, 0, {}), 2 * 3600,
                        dict(prep21, old_tokens=1000, new_tokens=900))
        check("counts unavailable and the estimate saves under 20%: not applied",
              len(got) == len(hist21), f"{st.get('bg_why')}")
        slow = lambda path, h, body: (time.sleep(5), (100, 200, {}))[1]
        t0 = time.time()
        got, st = run21("s21e", slow, 2 * 3600)
        check("a slow count endpoint costs at most the cap, then the estimate decides",
              time.time() - t0 < proxy.DECIDE_TIMEOUT_S + 1.5 and len(got) == 1,
              f"took {time.time() - t0:.1f}s")
        got, st = run21("s21f", by_len, 20 * 60)
        check("warm: decided on counted payback", ("repays in" in str(st.get("bg_why"))),
              f"{st.get('bg_why')}")
        P21 = "/Users/someone/Projects/Java/a-long-service-name/src/main/java/com/example"
        tail21 = [{"role": "assistant", "content": [{"type": "text", "text": f"ok {P21}/H{i}.java {P21}/K{i}.java"}]}
                  if i % 2 else {"role": "user", "content": [{"type": "text", "text": f"next {i}"}]}
                  for i in range(8)]
        saved_hist = hist21
        hist21 = old21 + tail21[:7]
        def counted(delta):
            return lambda path, h, body: ((len(json.dumps(body)) // 4) + (delta if "TokenSaver path aliases"
                                                                          in json.dumps(body) else 0), 200, {})
        run21("s21h", counted(-500), 2 * 3600)
        check("at the compaction, an alias plan that counts fewer tokens takes effect",
              (proxy._bg_load("s21h").get("aliases") or {}).get("mode") == "plan",
              "a measured saving would be thrown away")
        run21("s21i", counted(+500), 2 * 3600)
        check("at the compaction, a plan that counts more tokens is dropped",
              (proxy._bg_load("s21i").get("aliases") or {}).get("mode") == "off",
              "aliases that cost tokens would be kept")
        hist21 = saved_hist
        kinds = [x.get("type") for m in run21("s21g", by_len, 2 * 3600)[0] for x in m["content"]]
        check("an applied compaction sends no thinking from before it", "thinking" not in kinds,
              "the kept thinking would fail the prefix check")
    finally:
        proxy.audit._count, proxy.audit.enabled = saved_count, saved_enabled
        proxy.REWRITE, proxy.BG_COMPACT = "", False

    print("\n14. auxiliary requests that carry the main history share its rewrite")
    proxy.REWRITE = "forward"
    try:
        hist = body_obj["messages"][:3]
        h_main = dict(headers, **{"x-claude-code-session-id": "sess-aux"})
        for _s in ("sess-aux", "sess-aux-str"):   # seen before (else: adopted raw, section 22)
            proxy._bg_save(_s, {})
        post(base + "/v1/messages", json.dumps(dict(body_obj, messages=hist)).encode(), h_main)
        main_sent = json.loads(RECEIVED[-1]["body"])["messages"]
        aux = hist + [{"role": "user", "content": [{"type": "text", "text": "Suggest the next prompt."}]}]
        h_aux = dict(h_main, **{"x-claude-code-request-class": "auxiliary"})
        post(base + "/v1/messages", json.dumps(dict(body_obj, messages=aux)).encode(), h_aux)
        aux_sent = json.loads(RECEIVED[-1]["body"])["messages"]
        check("auxiliary request with the main history gets the same rewritten prefix",
              json.dumps(aux_sent[:2], sort_keys=True) == json.dumps(main_sent[:2], sort_keys=True)
              and aux_sent[:2] != hist[:2],
              "it would write its own copy of the whole history to the cache")
        # Claude Code's real shape: the newest message is a one-block list with
        # cache_control in the main request, and a plain string in the side request.
        h_s = dict(headers, **{"x-claude-code-session-id": "sess-aux-str"})
        newest = {"role": "user", "content": [{"type": "text", "text": "newest offset? what does that mean?",
                                               "cache_control": {"type": "ephemeral"}}]}
        post(base + "/v1/messages", json.dumps(dict(body_obj, messages=hist + [newest])).encode(), h_s)
        main_s = json.loads(RECEIVED[-1]["body"])["messages"]
        side = hist + [{"role": "user", "content": "newest offset? what does that mean?"}, aux[-1]]
        post(base + "/v1/messages", json.dumps(dict(body_obj, messages=side)).encode(),
             dict(h_s, **{"x-claude-code-request-class": "auxiliary"}))
        got = json.loads(RECEIVED[-1]["body"])["messages"]
        check("auxiliary request whose last main message is a plain string still matches",
              json.dumps(got[:len(hist)], sort_keys=True) == json.dumps(main_s[:len(hist)], sort_keys=True)
              and got[:len(hist)] != hist,
              "measured: 3 of 6 side requests a day missed on this alone, $3.5-4.4 each")
        other = [{"role": "user", "content": [{"type": "text", "text": "an unrelated history"}]}] + hist[1:]
        raw = json.dumps(dict(body_obj, messages=other + [aux[-1]])).encode()
        post(base + "/v1/messages", raw, h_aux)
        check("auxiliary request with a different history is relayed untouched",
              RECEIVED[-1]["body"] == raw, "the auto-mode classifier and probes must see original bytes")
        st_before = json.dumps(proxy._bg_load("sess-aux"), sort_keys=True)
        post(base + "/v1/messages", json.dumps(dict(body_obj, messages=aux)).encode(), h_aux)
        check("auxiliary requests do not touch session state",
              json.dumps(proxy._bg_load("sess-aux"), sort_keys=True) == st_before,
              "they would reset the idle timer that background compaction waits on")
    finally:
        proxy.REWRITE = ""

    print("\n22. history Anthropic saw raw stays raw (thinking is bound to it)")
    proxy.REWRITE = "forward"
    from tokensaver import routing as _routing
    saved_rs = _routing.state
    try:
        canon = lambda ms: json.dumps(proxy._canon(ms), sort_keys=True)
        script = "python3 - <<'PY'\n" + "print('step')\n" * 80 + "PY"

        def run(n):
            return {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": "sig" * 20},
                {"type": "tool_use", "id": f"t{n}", "name": "Bash", "input": {"command": script}}]}

        def res(n, text="done"):
            return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{n}", "content": text}]}
        ask = {"role": "user", "content": [{"type": "text", "text": "fix the build"}]}

        def send(msgs, sess, cls="main"):
            out, st = proxy._rewrite(json.dumps(dict(body_obj, messages=msgs)).encode(), "/v1/messages",
                                     cls, sess, dict(headers, **{"x-claude-code-session-id": sess}))
            return json.loads(out)["messages"], st

        # a session that started direct: the proxy's first sight of it already has replies
        h1 = [ask, run(1), res(1), run(2), res(2, "\x1b[31mred\x1b[0m   \nmore")]
        s1, st1 = send(h1, "s22-open")
        check("a session first seen mid-way sends its history through Claude's last reply unchanged",
              canon(s1[:4]) == canon(h1[:4]) and st1.get("adopted") == 4 and st1.get("adopted_why") == "first_seen",
              "every earlier thinking block fails the prefix check (old accounts drop it, new ones get a 400)")
        check("...and still trims what is new", "\x1b[31m" not in json.dumps(s1[4]),
              "no saving at all on sessions open at install")
        h1b = h1 + [{"role": "assistant", "content": [{"type": "text", "text": "fixed"}]},
                    {"role": "user", "content": [{"type": "text", "text": "thanks"}]}]
        s1b, _ = send(h1b, "s22-open")
        check("...and keeps sending those raw bytes on later requests", canon(s1b[:5]) == canon(s1[:5]),
              "the cache and the thinking break on the next request instead")
        fresh = [ask]
        send(fresh, "s22-new")
        s2, st2 = send([ask, run(1), res(1)], "s22-new")
        check("a session that started through the proxy is rewritten as before",
              "<<'PY'" not in json.dumps(s2[1]) and not st2.get("adopted"), "the saving would be lost for everyone")

        # routing restored by the watchdog after this session's last proxied request
        _routing.state = lambda: {"managed": True, "on": True, "at": time.time() + 5}
        h3 = [ask, run(1), res(1), run(2), res(2), run(3), res(3)]
        s3, st3 = send(h3, "s22-new")
        check("after a routing gap, the replies sent direct meanwhile stay raw",
              canon(s3[:2]) == canon(s2[:2]) and canon(s3[3:6]) == canon(h3[3:6]) and st3.get("adopted_why") == "gap",
              "the requests that went direct during the outage bound their thinking to the raw bytes")
        _routing.state = lambda: {"managed": True, "on": True, "at": time.time() - 3600}
        h4 = h3 + [run(4), res(4)]
        s4, st4 = send(h4, "s22-new")
        check("no gap: the newest reply is trimmed as usual",
              "<<'PY'" not in json.dumps(s4[7]) and not st4.get("adopted"), "a stale restore time would stop saving")

        # Claude Code's own /compact: same history, the instruction added to the newest message
        hc = [ask, run(1), res(1), {"role": "user", "content": [{"type": "text", "text": "go on"}]}]
        proxy._bg_save("s22-c", {})
        sc, _ = send(hc, "s22-c")
        compact = hc[:-1] + [{"role": "user", "content": [{"type": "text", "text": "go on"},
                                                          {"type": "text", "text": "Summarize the conversation."}]}]
        cc, stc = send(compact, "s22-c", "compaction")
        check("Claude Code's compaction request goes out with the main request's bytes",
              canon(cc[:3]) == canon(sc[:3]) and stc.get("auxiliary"),
              "it misses the cache and its thinking fails the prefix check (240 blocks in one /compact)")
        other = [{"role": "user", "content": [{"type": "text", "text": "another conversation"}]}] + hc[1:]
        raw = json.dumps(dict(body_obj, messages=other)).encode()
        out, _ = proxy._rewrite(raw, "/v1/messages", "compaction", "s22-c",
                                dict(headers, **{"x-claude-code-session-id": "s22-c"}))
        check("a compaction request for a different history is relayed untouched", out == raw,
              "someone else's bytes would be changed")
    finally:
        _routing.state = saved_rs
        proxy.REWRITE = ""

    px.shutdown(); up.shutdown()
    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all conformance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
