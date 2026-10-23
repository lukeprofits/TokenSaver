"""background: Claude Code's stale notices are trimmed in compaction, instructions are not."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokensaver import background as bg  # noqa: E402


def sysmsg(text):
    return {"role": "system", "content": [{"type": "text", "text": text}]}


class TrimTest(unittest.TestCase):
    def test_plan_mode_instructions_dropped_only_after_exit(self):
        on1, on2 = sysmsg("Plan mode is active. Do not edit."), sysmsg("Plan mode is active. Again.")
        msgs = [on1, sysmsg("## Exited Plan Mode\n\nYou have exited plan mode."), on2]
        stale = bg.stale_plan_mode(msgs)
        self.assertIn(id(on1), stale, "plan mode that ended is stale")
        self.assertNotIn(id(on2), stale, "plan mode still on is kept")
        out = "\n".join(bg.render(msgs, 300, skip=stale))
        self.assertNotIn("Do not edit", out)
        self.assertIn("Again", out)

    def test_file_change_notice_cut_like_a_tool_result(self):
        diff = "\n".join(f"   {i}\tline {i} of the file" for i in range(500))
        note = sysmsg(f"Note: /repo/src/A.java changed on disk since you last read it.\n{diff}")
        full = "\n".join(bg.render([note], 300))
        cut = "\n".join(bg.render([note], 300, drop_routine=True))
        self.assertGreater(len(full), 10_000)
        self.assertLess(len(cut), 2_000)
        self.assertIn("/repo/src/A.java", cut, "which file changed survives")

    def test_other_system_text_kept_verbatim(self):
        hook = sysmsg("SessionStart:startup hook success: respond tersely. " + "x" * 5000)
        self.assertIn("x" * 5000, "\n".join(bg.render([hook], 300, drop_routine=True)))

    def test_tools_loaded_in_compacted_part_stay_loaded(self):
        msgs = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "tool_reference", "tool_name": "WebFetch"},
                    {"type": "tool_reference", "tool_name": "Monitor"}]}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "x"}]}]
        self.assertEqual(bg.loaded_tools(msgs), ["Monitor", "WebFetch"])
        tools = [{"name": "Bash"}, {"name": "WebFetch", "defer_loading": True},
                 {"name": "Cron", "defer_loading": True}]
        out = bg.undefer(tools, ["WebFetch", "Monitor"])
        self.assertEqual(out, [{"name": "Bash"}, {"name": "WebFetch"}, {"name": "Cron", "defer_loading": True}])
        self.assertTrue(tools[1]["defer_loading"], "the caller's tools are not touched")
        self.assertIsNone(bg.undefer(tools, ["Bash"]), "nothing deferred: no change")
        self.assertIsNone(bg.undefer(tools, []))


if __name__ == "__main__":
    unittest.main()
