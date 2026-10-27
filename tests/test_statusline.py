"""The status-line badge: a missed check is not "down"; 20 s of misses is."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tokensaver-statusline"


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"ok": True, "rewrite": "forward", "fallback": False}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BadgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp() + "/"
        self.port = free_port()

    def badge(self):
        env = dict(os.environ, TMPDIR=self.tmp, TOKENSAVER_PROXY_PORT=str(self.port),
                   ANTHROPIC_BASE_URL=f"http://127.0.0.1:{self.port}")
        out = subprocess.run(["bash", str(SCRIPT)], input='{"session_id":"s1"}', env=env,
                             capture_output=True, text=True, timeout=20)
        return out.stdout

    def downfile(self) -> Path:
        return Path(self.tmp) / f"tokensaver-health-{os.getuid()}-{self.port}.down"

    def test_one_missed_check_is_not_down(self):
        self.assertNotIn("down", self.badge(), "a fresh miss must not alarm anyone")
        self.assertIn("⚡", self.badge())

    def test_twenty_seconds_of_misses_is_down(self):
        self.badge()
        self.downfile().write_text(str(int(time.time()) - 25))
        self.assertIn("down", self.badge())

    def test_a_healthy_proxy_clears_it_at_once(self):
        self.badge()
        self.downfile().write_text(str(int(time.time()) - 25))
        srv = ThreadingHTTPServer(("127.0.0.1", self.port), Health)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            out = self.badge()
            self.assertIn("⚡", out)
            self.assertNotIn("down", out, "a failure must never be remembered once the proxy answers")
            self.assertFalse(self.downfile().exists())
        finally:
            srv.shutdown()
            srv.server_close()

    def test_an_empty_cache_from_the_old_script_is_checked_again(self):
        cache = Path(self.tmp) / f"tokensaver-health-{os.getuid()}-{self.port}"
        cache.write_text("")                       # what the old script left after one miss
        srv = ThreadingHTTPServer(("127.0.0.1", self.port), Health)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            self.assertNotIn("down", self.badge())
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
