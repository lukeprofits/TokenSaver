"""Long inline scripts are left out of the history copy behind a path that cannot run: a copy
Claude makes of the short form fails, never runs old code, never truncates a file. Nothing is
written to disk."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tokensaver import passes

BODY = "print('hello from the real script')\n" * 30           # > 600 chars


class OmitTest(unittest.TestCase):
    def test_shapes(self):
        cmd = f"cd /tmp && python3 - <<'PY'\n{BODY}PY"
        short = passes.omit_heredocs(cmd)
        self.assertTrue(short.startswith("cd /tmp && python3 /dev/null/tokensaver-omitted/"))
        self.assertNotIn("hello", short)
        w = passes.omit_heredocs(f"cat > out.txt <<'EOF'\n{BODY}EOF")
        self.assertTrue(w.startswith("false && cat /dev/null/tokensaver-omitted/"))
        self.assertTrue(w.endswith("> out.txt"))
        a = passes.omit_heredocs(f"cat >> log.txt <<'EOF'\n{BODY}EOF\necho done")
        self.assertIn(">> log.txt", a)
        self.assertTrue(a.endswith("echo done"), "the rest of the chain stays")

    def test_left_alone(self):
        self.assertIsNone(passes.omit_heredocs("python3 - <<'PY'\nprint(1)\nPY"), "short bodies stay")
        self.assertIsNone(passes.omit_heredocs(f"python3 - <<PY\n{BODY}PY"), "unquoted tag: the shell expands it")
        self.assertIsNone(passes.omit_heredocs("ls -la"))

    def test_deterministic(self):
        cmd = f"python3 - <<'PY'\n{BODY}PY"
        self.assertEqual(passes.omit_heredocs(cmd), passes.omit_heredocs(cmd))

    def run_in_shell(self, command, cwd):
        return subprocess.run(["/bin/sh", "-c", command], cwd=cwd, capture_output=True, text=True, timeout=30)

    def test_a_copy_of_the_short_form_cannot_run_or_write(self):
        d = Path(tempfile.mkdtemp())
        target = d / "out.txt"
        target.write_text("precious\n")
        script = passes.omit_heredocs(f"python3 - <<'PY'\n{BODY}PY")
        r = self.run_in_shell(script, d)
        self.assertNotEqual(r.returncode, 0, "it fails")
        self.assertNotIn("hello from the real script", r.stdout, "nothing ran")
        write = passes.omit_heredocs(f"cat > out.txt <<'EOF'\n{BODY}EOF")
        r = self.run_in_shell(write, d)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(target.read_text(), "precious\n", "the target is not truncated")
        append = passes.omit_heredocs(f"cat >> out.txt <<'EOF'\n{BODY}EOF")
        self.run_in_shell(append, d)
        self.assertEqual(target.read_text(), "precious\n")

    def test_the_pass_writes_no_file_and_passes_its_check(self):
        before = {p for p in Path(os.environ["HOME"]).rglob("*")}
        msgs = [{"role": "user", "content": [{"type": "text", "text": "run it"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                                   "input": {"command": f"python3 - <<'PY'\n{BODY}PY",
                                                             "description": "run the script"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]
        out = passes.rewrite_messages(msgs)
        inp = out[1]["content"][0]["input"]
        self.assertIn("/dev/null/tokensaver-omitted/", inp["command"])
        self.assertIn("cannot run", inp["description"])
        self.assertTrue(inp["description"].startswith("run the script"))
        self.assertEqual(msgs[1]["content"][0]["input"]["command"].count("hello"), 30, "original untouched")
        self.assertEqual({p for p in Path(os.environ["HOME"]).rglob("*")}, before, "no file written")


class CleanupAndClosedTest(unittest.TestCase):
    def test_old_scripts_are_deleted_by_setup(self):
        from unittest import mock
        from tokensaver import setup
        d = Path(tempfile.mkdtemp())
        (d / "scripts").mkdir()
        (d / "scripts" / "abc.py").write_text("print('old')")
        with mock.patch.object(setup.vault, "VAULT_DIR", d):
            step = setup.old_scripts_step()
            self.assertIn("1 old saved scripts", step.detail)
            step.apply()
            self.assertFalse((d / "scripts").exists())
            self.assertIsNone(setup.old_scripts_step(), "nothing left: no step")

    def test_a_cleared_session_gets_no_bumps_after_its_end(self):
        from tokensaver import keepalive as ka
        d = Path(tempfile.mkdtemp())
        p = d / "s.jsonl"
        p.write_text('{"type":"user","message":{"role":"user","content":"<command-name>/clear</command-name>"}}\n')
        self.assertTrue(ka.closed_by_user(p))
        p.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
        self.assertFalse(ka.closed_by_user(p))


if __name__ == "__main__":
    unittest.main()
