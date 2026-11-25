"""Script shortening (the heredocs pass) runs only for Opus and newer by default."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import json
import unittest
import uuid

from tokensaver import config, passes, proxy

SCRIPT = "python3 - <<'PY'\n" + "print('line of a long inline script')\n" * 40 + "PY"


def _body(model: str) -> bytes:
    return json.dumps({"model": model, "max_tokens": 100, "messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Bash",
             "input": {"command": SCRIPT, "description": "edit the file"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
    ]}).encode()


def _sent_command(model: str) -> str:
    session = "ss-" + uuid.uuid4().hex[:8]
    # A session's first request is seen with no history of its own (history the proxy never
    # sent goes out as it came), then the next one carries the script.
    first = json.dumps({"model": model, "max_tokens": 100,
                        "messages": [{"role": "user", "content": "fix it"}]}).encode()
    proxy._rewrite(first, "/v1/messages", "main", session, {})
    out, _ = proxy._rewrite(_body(model), "/v1/messages", "main", session, {})
    return json.loads(out)["messages"][1]["content"][0]["input"]["command"]


class RuleTest(unittest.TestCase):
    def test_auto_by_model(self):
        self.assertTrue(passes.scripts_on("claude-opus-5-5"))
        self.assertTrue(passes.scripts_on("claude-fable-5-1"))
        for model in ("claude-sonnet-5-5", "claude-haiku-4-5-20251001", "", None):
            self.assertFalse(passes.scripts_on(model), model)

    def test_setting_wins(self):
        self.assertTrue(passes.scripts_on("claude-sonnet-5-5", "on"))
        self.assertFalse(passes.scripts_on("claude-opus-5-5", "off"))


class ProxyTest(unittest.TestCase):
    def setUp(self):
        self.old = proxy.REWRITE
        proxy.REWRITE = "forward"

    def tearDown(self):
        proxy.REWRITE = self.old
        config.unset("script_shortening")

    def test_opus_shortened_sonnet_not(self):
        self.assertIn(passes.OMITTED_DIR, _sent_command("claude-opus-5-5"))
        self.assertEqual(_sent_command("claude-sonnet-5-5"), SCRIPT)

    def test_setting_reaches_the_proxy(self):
        config.set_value("script_shortening", "on")
        self.assertIn(passes.OMITTED_DIR, _sent_command("claude-sonnet-5-5"))
        config.set_value("script_shortening", "off")
        self.assertEqual(_sent_command("claude-opus-5-5"), SCRIPT)


if __name__ == "__main__":
    unittest.main()
