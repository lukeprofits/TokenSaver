"""Local AI: which machine gets which model, the background download (resume,
checksum, one at a time), the toggle, and `--benchmark full`'s guard and estimate.
No network beyond a stub server on 127.0.0.1, no real ollama, no services."""

import tests  # noqa: F401  isolation first (tests/__init__.py): fake HOME, guarded commands

import hashlib
import http.server
import io
import json
import shutil
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tokensaver import benchmark, config, localai, machine, proxy

GB = 2 ** 30


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved_cfg = config.PATH
        config.PATH = self.tmp / "config.json"
        config.load(force=True)
        self.saved = {k: getattr(localai, k) for k in ("DIR", "BIN", "MODELS", "STATE", "LOCK")}
        localai.DIR = self.tmp / "ollama"
        localai.BIN = localai.DIR / "bin" / "ollama"
        localai.MODELS = localai.DIR / "models"
        localai.STATE = self.tmp / "local_ai.json"
        localai.LOCK = self.tmp / "local_ai.lock"

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(localai, k, v)
        config.PATH = self.saved_cfg
        config.load(force=True)

    def decide(self, plat="macos", mem=64, disk=500, tags=None, arch="arm64", mode=None, measured=None,
               own_tested=True):
        with mock.patch.object(localai, "_platform_name", return_value=plat), \
                mock.patch.dict(localai.OWN_OLLAMA_TESTED, {plat: own_tested}), \
                mock.patch.object(localai, "_arch", return_value=arch), \
                mock.patch.object(machine, "model_memory_bytes", return_value=mem * GB), \
                mock.patch.object(machine, "disk_free_bytes", return_value=disk * GB), \
                mock.patch.object(machine, "gpu", return_value=("nvidia", mem * GB) if plat == "linux" else (None, 0)), \
                mock.patch.object(localai, "_tags", return_value=tags), \
                mock.patch.object(localai, "_can_unzstd", return_value=True), \
                mock.patch.dict(localai.LINUX_MEASURED, measured or {}):
            return localai.decide(mode)


class DecideTest(Base):
    def tags(self, d):
        return [t for _, t, _ in d["models"]]

    def test_mac_tiers_by_memory(self):
        self.assertIn("16 GB memory; needs 24 GB+", self.decide(mem=16)["skip"])
        self.assertEqual(self.tags(self.decide(mem=32)), ["nimble"])
        self.assertEqual(self.tags(self.decide(mem=64)), ["nimble", "qwen3.6:35b-mlx"])
        self.assertIn("Apple-silicon", self.decide(arch="amd64")["skip"])

    def test_linux_needs_a_card_and_measured_builds(self):
        self.assertIn("graphics card", self.decide(plat="linux", mem=0)["skip"])
        self.assertIn("not available on Linux yet", self.decide(plat="linux", mem=48)["skip"])
        both = {"judge": True, "check": True}
        self.assertEqual(self.tags(self.decide(plat="linux", mem=16, measured=both)), ["nimble:9b-q8_0"])
        self.assertEqual(self.tags(self.decide(plat="linux", mem=48, measured=both)),
                         ["nimble:9b-q8_0", "qwen3.6:35b"])
        self.assertIn("12 GB graphics-card memory", self.decide(plat="linux", mem=12, measured=both)["skip"])

    def test_disk_check(self):
        d = self.decide(disk=99)
        self.assertIn("99 GB free disk; needs more than 100 GB", d["skip"])
        self.assertIsNone(self.decide(disk=101)["skip"])
        self.assertIsNone(self.decide(disk=10, mode="on")["skip"], "`on` skips the disk check")
        self.assertIn("16 GB", self.decide(mem=16, mode="on")["skip"], "never the specs check")

    def test_existing_ollama_with_models_needs_nothing(self):
        d = self.decide(disk=5, tags=["nimble:latest", "qwen3.6:35b-mlx"])
        self.assertEqual((d["ollama"], d["missing"], d["skip"]), ("system", [], None),
                         "models already there: no download, so no disk check")
        d = self.decide(tags=["nimble:latest"])
        self.assertEqual((d["ollama"], d["missing"]), ("system", ["qwen3.6:35b-mlx"]))
        self.assertEqual(d["need_bytes"], localai.TIERS["macos"][1][2], "models only: no ollama download")
        d = self.decide(tags=None)
        self.assertEqual(d["ollama"], "ours")
        self.assertGreater(d["need_bytes"], sum(b for _, _, b, _ in localai.TIERS["macos"]))

    def test_own_ollama_held_back_until_tested(self):
        d = self.decide(tags=None, own_tested=False)
        self.assertIn("needs ollama for now", d["skip"])
        self.assertEqual(d["need_bytes"], 0, "nothing is downloaded")
        d = self.decide(tags=["nimble:latest"], own_tested=False)
        self.assertIsNone(d["skip"], "a system ollama is used as before")

    def test_off(self):
        d = self.decide(mode="off")
        self.assertIn("turned it off", d["skip"])
        self.assertEqual(d["models"], [])

    def test_configure_uses_tuned_slot_and_respects_the_user(self):
        d = self.decide(plat="linux", mem=48, measured={"judge": True, "check": True}, tags=[])
        changed = localai.configure(d)
        self.assertIn("prune_judge", changed)
        self.assertEqual(config.get("prune_judge"), "nimble:9b-q8_0")
        self.assertEqual(config.get("ollama_url"), localai.SYSTEM_URL + "/api/generate")
        config.set_value("local_model", "my-own", "user")
        self.assertNotIn("local_model", localai.configure(d), "a user's own setting wins")
        self.assertEqual(config.get("local_model"), "my-own")

    def test_describe(self):
        d = self.decide(tags=["nimble:latest", "qwen3.6:35b-mlx"])
        self.assertEqual(localai.describe(d), "on (Nimble + 35B, using your ollama)")
        self.assertTrue(localai.describe(self.decide(mem=16)).startswith("skipped (16 GB memory"))
        localai._save_state({"pid": None, "error": "network down"})
        self.assertIn("retries by itself", localai.describe(self.decide(tags=[])))


class Stub(http.server.BaseHTTPRequestHandler):
    files: dict = {}
    pulls: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            return self._send(json.dumps({"models": [{"name": t} for t in Stub.pulls]}).encode())
        name = self.path.rsplit("/", 1)[-1]
        data = Stub.files.get(name)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])
            return
        self._send(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.pulls.append(body["model"])
        lines = [{"status": "pulling", "digest": "sha256:a", "total": 100, "completed": 50},
                 {"status": "pulling", "digest": "sha256:a", "total": 100, "completed": 100},
                 {"status": "success"}]
        self._send(b"".join(json.dumps(x).encode() + b"\n" for x in lines))

    def _send(self, data):
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def tgz(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class SyncTest(Base):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        super().setUp()
        archive = tgz({"ollama": b"#!/bin/sh\n", "libggml.dylib": b"x"})
        Stub.files = {"ollama-darwin.tgz": archive,
                      "sha256sum.txt": f"{hashlib.sha256(archive).hexdigest()}  ./ollama-darwin.tgz\n".encode()}
        Stub.pulls = []
        self.patches = [mock.patch.object(localai, "RELEASE", self.url + "/"),
                        mock.patch.object(localai, "own_url", return_value=self.url),
                        mock.patch.object(localai, "_platform_name", return_value="macos"),
                        mock.patch.object(localai, "_arch", return_value="arm64"),
                        mock.patch.object(machine, "model_memory_bytes", return_value=64 * GB),
                        mock.patch.object(machine, "disk_free_bytes", return_value=500 * GB),
                        mock.patch.object(localai, "SYSTEM_URL", "http://127.0.0.1:1"),   # nothing there
                        mock.patch("tokensaver.service.install_ollama"),
                        mock.patch.object(localai.vault, "record"),
                        mock.patch.dict(localai.OWN_OLLAMA_TESTED, {"macos": True})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        super().tearDown()

    def test_installs_ollama_then_the_models(self):
        self.assertEqual(localai.sync(log=lambda m: None), 0)
        self.assertTrue(localai.BIN.exists(), "binary unpacked into ~/.tokensaver/ollama/bin")
        self.assertEqual(Stub.pulls, ["nimble", "qwen3.6:35b-mlx"])
        st = localai.load_state()
        self.assertTrue(st["finished"])
        self.assertEqual(st["pulled"], ["nimble", "qwen3.6:35b-mlx"])
        self.assertEqual(config.get("ollama_url"), self.url + "/api/generate")
        self.assertFalse((localai.DIR / "download").exists(), "downloads cleaned up")

    def test_bad_checksum_stops_and_deletes(self):
        Stub.files["sha256sum.txt"] = b"0" * 64 + b"  ./ollama-darwin.tgz\n"
        self.assertEqual(localai.sync(log=lambda m: None), 1)
        self.assertFalse(localai.BIN.exists())
        self.assertIn("checksum", localai.load_state()["error"])
        self.assertFalse((localai.DIR / "download" / "ollama-darwin.tgz").exists())
        self.assertEqual(Stub.pulls, [], "no model download after a failed ollama install")

    def test_download_resumes(self):
        dest = self.tmp / "a.tgz"
        data = Stub.files["ollama-darwin.tgz"]
        dest.with_name("a.tgz.part").write_bytes(data[:10])
        localai._fetch(self.url + "/ollama-darwin.tgz", dest)
        self.assertEqual(dest.read_bytes(), data)

    def test_one_at_a_time(self):
        held = localai._lock()
        try:
            self.assertEqual(localai.sync(log=lambda m: None), 0)
            self.assertEqual(Stub.pulls, [], "a second sync does nothing while one runs")
        finally:
            held.close()


class SafetyTest(Base):
    def test_archive_paths_stay_inside(self):
        bad = self.tmp / "bad.tgz"
        bad.write_bytes(tgz({"../escape": b"x"}))
        with self.assertRaises(RuntimeError):
            localai._unpack(bad, self.tmp / "out")
        self.assertFalse((self.tmp / "escape").exists())

    def test_checksum_list_parsing(self):
        sums = "abc  ./ollama-darwin.tgz\ndef  ollama-linux-amd64.tar.zst\n"
        self.assertEqual(localai._expected(sums, "ollama-darwin.tgz"), "abc")
        self.assertEqual(localai._expected(sums, "ollama-linux-amd64.tar.zst"), "def")
        self.assertIsNone(localai._expected(sums, "other"))


class ToggleTest(Base):
    def test_watchdog_restarts_an_unfinished_download(self):
        localai._save_state({"want": ["nimble"], "finished": False, "started": time.time() - 3600})
        with mock.patch.object(localai, "start", return_value=True) as start:
            self.assertEqual(localai.kick(), "local AI download restarted")
            start.assert_called_once()
        localai._save_state({"want": ["nimble"], "finished": False, "started": time.time()})
        with mock.patch.object(localai, "start", return_value=True) as start:
            self.assertIsNone(localai.kick(), "not again within RETRY_S")
            start.assert_not_called()
        localai._save_state({"want": ["nimble"], "finished": True})
        self.assertIsNone(localai.kick())

    def test_off_stops_and_keeps_files(self):
        localai.MODELS.mkdir(parents=True)
        (localai.MODELS / "blob").write_bytes(b"x" * 1000)
        with mock.patch("tokensaver.service.remove_ollama", return_value=True):
            msg = localai.set_mode("off")
        self.assertEqual(config.get("local_ai"), "off")
        self.assertIn("our ollama stopped", msg)
        self.assertTrue((localai.MODELS / "blob").exists(), "off keeps the files")
        self.assertTrue(localai.load_state()["finished"])

    def test_uninstall_removes_ours_unless_kept(self):
        localai.MODELS.mkdir(parents=True)
        with mock.patch("tokensaver.service.remove_ollama", return_value=True):
            kept = localai.remove(keep_files=True)
            self.assertTrue(localai.DIR.exists())
            gone = localai.remove()
        self.assertIn("stopped TokenSaver's ollama", kept)
        self.assertFalse(localai.DIR.exists())
        self.assertTrue(any("deleted" in x for x in gone))


class BenchmarkModeTest(unittest.TestCase):
    def test_full_refuses_without_local_ai(self):
        with mock.patch("tokensaver.prune.tier", return_value=("off", "no models")), \
                mock.patch.object(benchmark.sim, "sessions", return_value=[]), \
                mock.patch("tokensaver.localai.describe", return_value="skipped (test)"):
            self.assertEqual(benchmark.run(mode="full", yes=True), 2)

    def test_estimate_counts_long_breaks_with_context(self):
        R = lambda t, ctx: SimpleNamespace(timestamp=t, context=ctx)
        d = SimpleNamespace(requests=[R("2026-10-01T10:00:00Z", 200_000), R("2026-10-01T12:00:00Z", 200_000),
                                      R("2026-10-01T12:01:00Z", 200_000), R("2026-10-01T15:00:00Z", 20_000)])
        n, secs = benchmark.full_estimate([d], 15 * 60, "deep")
        self.assertEqual(n, 1, "one long break with enough context; the small one is not counted")
        self.assertEqual(secs, benchmark.SECONDS_PER_BREAK["deep"])
        self.assertIn("hours", benchmark._hours(9 * 3600))
        self.assertIn("minutes", benchmark._hours(600))


class PruneOfflineTest(unittest.TestCase):
    def test_restores_the_proxy_and_returns_rules_only_when_off(self):
        path, record = proxy._bg_path, proxy.vault.record
        pending = {"cut": 2, "text": "x", "new_tokens": 10}
        with mock.patch("tokensaver.prune.tier", return_value=("off", "test")):
            got, events = proxy.prune_offline("s", [], pending, Path(tempfile.mkdtemp()))
        self.assertEqual(got["text"], "x")
        self.assertIs(proxy._bg_path, path)
        self.assertIs(proxy.vault.record, record)


if __name__ == "__main__":
    unittest.main()
