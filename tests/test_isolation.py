"""Fails the run if any tokensaver path constant still points at the real home."""

import importlib
import pkgutil
import unittest
from pathlib import Path

import tokensaver
from tests import FAKE_HOME, REAL_HOME

REPO = Path(tokensaver.__file__).resolve().parent.parent


def path_constants():
    """(module.NAME, path) for every module-level Path, or str that looks like a path."""
    out = []
    for info in pkgutil.iter_modules(tokensaver.__path__):
        mod = importlib.import_module(f"tokensaver.{info.name}")
        for name, value in vars(mod).items():
            if name.startswith("__"):
                continue
            if isinstance(value, Path) or (isinstance(value, str) and value.startswith("/")):
                out.append((f"{info.name}.{name}", Path(value)))
    return out


def under(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class IsolationTest(unittest.TestCase):
    def test_no_constant_points_at_the_real_home(self):
        found = path_constants()
        self.assertGreater(len(found), 10, "the scan found almost nothing: it is broken")
        bad = [f"{n} = {p}" for n, p in found
               if under(p, REAL_HOME) and not under(p, REPO) and not under(p, FAKE_HOME)]
        self.assertEqual(bad, [], "these would touch the real machine in tests")

    def test_home_is_the_fake_one(self):
        self.assertEqual(Path.home(), FAKE_HOME)


if __name__ == "__main__":
    for n, p in sorted(path_constants()):
        print(f"{n:32s} {p}")
