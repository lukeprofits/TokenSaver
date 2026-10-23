
import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands
import json
import tempfile
import unittest
from pathlib import Path

from tokensaver import benchmark, transcripts


def rec(role, content, mid=None, side=False, usage=True, **kw):
    m = {"role": role, "content": content}
    if mid:
        m["id"], m["model"] = mid, "claude-opus-5-5"
        if usage:
            m["usage"] = {"input_tokens": 10, "cache_read_input_tokens": 100, "output_tokens": 5}
    return dict({"message": m, "isSidechain": side, "timestamp": "2026-10-22T10:00:00Z"}, **kw)


class HistoryTest(unittest.TestCase):
    def write(self, records):
        p = Path(tempfile.mkdtemp()) / "s.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records))
        return p

    def test_sidechain_left_out_and_requests_line_up(self):
        p = self.write([rec("user", "hi"), rec("assistant", [{"type": "text", "text": "a"}], "m1"),
                        rec("user", "sub", side=True), rec("assistant", "x", "s1", side=True),
                        rec("user", "next"), rec("assistant", [{"type": "text", "text": "b"}], "m2")])
        msgs, reqs = transcripts.api_history(p)
        self.assertEqual(len(reqs), 2, "subagent request not counted")
        self.assertEqual([m["content"] for m in transcripts.request_messages(msgs, reqs[1])][-1], [{"type": "text", "text": "next"}])

    def test_history_restarts_at_claude_codes_compaction(self):
        p = self.write([rec("user", "old"), rec("assistant", "a", "m1"),
                        {"subtype": "compact_boundary", "compactMetadata": {}},
                        rec("user", "summary"), rec("assistant", "b", "m2")])
        msgs, reqs = transcripts.api_history(p)
        sent = transcripts.request_messages(msgs, reqs[1])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["content"][0]["text"], "summary")

    def test_tool_references_become_text_for_counting(self):
        m = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                           "content": [{"type": "tool_reference", "tool_name": "WebSearch"}]}]}]
        out = benchmark._countable(m)
        self.assertEqual(out[0]["content"][0]["content"][0], {"type": "text", "text": "[tool WebSearch]"})
        self.assertEqual(m[0]["content"][0]["content"][0]["type"], "tool_reference", "original untouched")


if __name__ == "__main__":
    unittest.main()
