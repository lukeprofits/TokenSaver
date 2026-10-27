"""Rebuilding the API history from a transcript: parallel tool calls stay one reply."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import json
import tempfile
import unittest
from pathlib import Path

from tokensaver import transcripts


def rec(role, content, mid=None, usage=False):
    msg = {"role": role, "content": content}
    if mid:
        msg["id"] = mid
    if usage:
        msg["usage"] = {"input_tokens": 3, "cache_read_input_tokens": 100, "output_tokens": 5}
        msg["model"] = "claude-opus-5-5"
    return {"type": role, "message": msg, "timestamp": "2026-10-27T10:00:00Z"}


def call(i):
    return [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"echo {i}"}}]


def result(i):
    return [{"type": "tool_result", "tool_use_id": f"t{i}", "content": str(i)}]


class ParallelToolsTest(unittest.TestCase):
    def history(self, records):
        p = Path(tempfile.mkdtemp()) / "s.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        return transcripts.api_history(p)

    def test_results_written_mid_reply_follow_the_whole_reply(self):
        # Claude Code's file order for one reply with three tool calls (measured):
        # call 1, call 2, result 1, call 3, result 2, result 3.
        msgs, entries = self.history([
            rec("user", "run three things"),
            rec("assistant", call(1), "msg_a", usage=True), rec("assistant", call(2), "msg_a"),
            rec("user", result(1)), rec("assistant", call(3), "msg_a"),
            rec("user", result(2)), rec("user", result(3)),
            rec("assistant", [{"type": "text", "text": "done"}], "msg_b", usage=True),
        ])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant"])
        self.assertEqual([b["id"] for b in msgs[1]["content"]], ["t1", "t2", "t3"])
        self.assertEqual([b["tool_use_id"] for b in msgs[2]["content"]], ["t1", "t2", "t3"])
        self.assertEqual([e[0] for e in entries], [1, 3], "each request still sends what came before it")
        self.assertTrue(all(set(m) == {"role", "content"} for m in msgs), "no bookkeeping field may reach the API")

    def test_a_new_reply_after_results_is_not_merged_back(self):
        msgs, _ = self.history([
            rec("user", "go"), rec("assistant", call(1), "msg_a", usage=True), rec("user", result(1)),
            rec("assistant", call(2), "msg_b", usage=True), rec("user", result(2)),
        ])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant", "user"])

    def test_user_text_between_keeps_the_old_order(self):
        msgs, _ = self.history([
            rec("user", "go"), rec("assistant", call(1), "msg_a", usage=True),
            rec("user", [{"type": "text", "text": "wait"}]), rec("assistant", call(2), "msg_a"),
        ])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant"])


if __name__ == "__main__":
    unittest.main()
