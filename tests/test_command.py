"""The `tokensaver` terminal command: link + PATH block, idempotent, never someone else's file."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokensaver import __main__ as cli, setup


class CommandStepTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.rc = self.home / ".zshrc"
        self.rc.write_text("export A=1\n")
        self.p = [mock.patch.object(setup, "BIN_DIR", self.home / ".local" / "bin"),
                  mock.patch.object(setup, "rc_file", return_value=self.rc),
                  mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"})]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()

    def test_link_and_path_block_once(self):
        step = setup.command_step()
        self.assertEqual(step.status, "todo")
        step.apply()
        link = self.home / ".local" / "bin" / "tokensaver"
        self.assertTrue(link.is_symlink() and os.path.realpath(link) == os.path.realpath(setup.CLI))
        text = self.rc.read_text()
        self.assertTrue(text.startswith("export A=1\n"))
        self.assertEqual(text.count(setup.PATH_LINE), 1)
        self.assertEqual(setup.command_step().status, "already", "second run changes nothing")

    def test_no_path_block_when_already_on_path(self):
        with mock.patch.dict(os.environ, {"PATH": f"{self.home / '.local' / 'bin'}:/usr/bin"}):
            setup.command_step().apply()
        self.assertEqual(self.rc.read_text(), "export A=1\n")

    def test_old_alias_block_replaced(self):
        self.rc.write_text(f"export A=1\n\n{setup.RC_BEGIN}\nalias claude-ts='x'\n{setup.RC_END}\n")
        step = setup.command_step()
        self.assertTrue(any("claude-ts" in n for n in step.notes))
        step.apply()
        self.assertNotIn("claude-ts", self.rc.read_text())

    def test_someone_elses_tokensaver_is_left_alone(self):
        d = self.home / ".local" / "bin"
        d.mkdir(parents=True)
        (d / "tokensaver").write_text("#!/bin/sh\necho other\n")
        self.assertEqual(setup.command_step().status, "skip")
        self.assertEqual((d / "tokensaver").read_text(), "#!/bin/sh\necho other\n")


class FlagTest(unittest.TestCase):
    def test_benchmark_flag_is_the_benchmark_command(self):
        from tokensaver import benchmark
        with mock.patch.object(benchmark, "run", return_value=0) as run:
            self.assertEqual(cli.main(["--benchmark", "--yes"]), 0)
        self.assertTrue(run.called and run.call_args.kwargs["yes"])


if __name__ == "__main__":
    unittest.main()
