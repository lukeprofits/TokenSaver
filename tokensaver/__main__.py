"""TokenSaver - CLI entry point. Most commands only read local files; `setup`,
`install`, `uninstall`, `config set` and `restart` change things, and say so."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import (__version__, doctor, install, proxy, report, sessions, simulate,
               vault)
from .transcripts import DEFAULT_ROOT, load

DEFAULT_THRESHOLDS = [400_000, 300_000, 200_000, 100_000]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tokensaver", description=__doc__)
    ap.add_argument("--version", action="version", version=f"TokenSaver {__version__}")
    # Shared by every transcript-reading subcommand. Deliberately NOT also on
    # the top-level parser: argparse lets a subparser's defaults clobber a
    # parent value, so `--since` before the subcommand parsed cleanly and then
    # silently did nothing. One position, one meaning.
    src = argparse.ArgumentParser(add_help=False)
    src.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                     help=f"transcript directory (default: {DEFAULT_ROOT})")
    src.add_argument("--since", metavar="YYYY-MM-DD",
                     help="ignore sessions ending before this date")
    src.add_argument("--min-requests", type=int, default=1, metavar="N",
                     help="ignore sessions with fewer than N billed requests")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("report", parents=[src], help="measured baseline: tokens sent and what they cost")
    p_rep.add_argument("--reprice-as", metavar="MODEL",
                       help="price every request at this model's rates, e.g. claude-opus-5-5")

    sub.add_parser("doctor", parents=[src], help="what is configured, what it resolved to, what is assumed")

    p_ses = sub.add_parser("sessions", parents=[src], help="per-session cost, highest first")
    p_ses.add_argument("--limit", type=int, default=15)

    p_sim = sub.add_parser("simulate", parents=[src], help="replay sessions under compaction thresholds")
    p_sim.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS,
                       metavar="N", help="autoCompactWindow values to test")
    p_sim.add_argument("--reprice-as", metavar="MODEL")
    p_sim.add_argument("--rebuild-as", choices=("write", "input"), default="write",
                       help="how the resumed prefix bills after a compaction (default: write, "
                            "the pessimistic reading)")
    p_sim.add_argument("--post-tokens", type=int, default=simulate.MEASURED_POST_TOKENS,
                       metavar="N", help="post-compaction seed size")

    p_search = sub.add_parser("search", help="full-text search the vault (exact stored bytes)")
    p_search.add_argument("query")
    p_search.add_argument("--kind")
    p_search.add_argument("--limit", type=int, default=20)

    p_read = sub.add_parser("read", help="print one event's exact stored content")
    p_read.add_argument("event_id", type=int)
    p_read.add_argument("--head", type=int, metavar="N", help="first N characters only")
    p_read.add_argument("--tail", type=int, metavar="N", help="last N characters only")

    sub.add_parser("vault", help="vault statistics")

    sub.add_parser("install", help="add the vault hooks to Claude Code settings")
    p_un = sub.add_parser("uninstall", help="remove TokenSaver completely (Claude Code goes back to direct)")
    p_un.add_argument("--yes", action="store_true", help="do not ask (keeps your data unless --purge)")
    g_un = p_un.add_mutually_exclusive_group()
    g_un.add_argument("--purge", action="store_true", help="also delete ~/.tokensaver (vault, settings, logs)")
    g_un.add_argument("--keep-data", action="store_true", help="keep ~/.tokensaver without asking")

    p_prox = sub.add_parser("proxy", help="the gateway: relay API traffic, rewriting only what stays byte-stable")
    p_prox.add_argument("--port", type=int, default=proxy.PORT)
    p_prox.add_argument("--upstream", default=proxy.UPSTREAM)
    p_prox.add_argument("--stats", action="store_true",
                        help="report what the relay saw, including the cache check")
    p_prox.add_argument("--reset-fallback", action="store_true",
                        help="clear an automatic fallback so rewriting resumes")
    p_prox.add_argument("--service", action="store_true",
                        help="run as the background service (launchd): wait on a busy port, drain on stop")

    p_setup = sub.add_parser("setup", help="install TokenSaver for this machine (shows the plan, asks once)")
    p_setup.add_argument("--dry-run", action="store_true")
    p_setup.add_argument("--yes", action="store_true", help="apply without asking")
    p_setup.add_argument("--refresh", action="store_true", help=argparse.SUPPRESS)   # used by `update`
    p_setup.add_argument("--route", choices=("settings", "alias"), default="settings",
                         help="settings (default): every Claude Code launch, terminal and IDE, goes through "
                              "tokensaver; alias: only shell aliases (claude-ts), as before")
    p_setup.add_argument("--shell-rc", choices=("yes", "no"), default="yes",
                         help="with --route alias: edit the shell rc (default) or print the aliases")
    p_setup.add_argument("--audit", choices=("on", "off"))
    p_setup.add_argument("--no-service", action="store_true")
    p_setup.add_argument("--with-hooks", action="store_true", help="also install vault hooks")
    p_setup.add_argument("--statusbar", choices=("yes", "no"), default="yes",
                         help="show ⚡ and the % saved in Claude Code's status line (default yes)")
    p_setup.add_argument("--with-prune", action="store_true",
                         help="also set up local models (ollama) that prune each compaction a little further "
                              "(~0.5-1%% more): Nimble ~9 GB, 24 GB+; on a 48 GB+ Mac also the 35B ~22 GB")
    p_setup.add_argument("--aliases", choices=("try", "replace"), default="try",
                         help="try: add claude-ts beside your aliases (default); replace: reroute claude itself")
    sub.add_parser("restart", help="restart the background proxy once nothing is in flight")
    sub.add_parser("update", help="pull the latest TokenSaver and restart the proxy")
    p_mig = sub.add_parser("migrate", help="move this machine from the old name (everytoken) to TokenSaver, once")
    p_mig.add_argument("--dry-run", action="store_true")
    p_mig.add_argument("--yes", action="store_true")
    sub.add_parser("watchdog", help=argparse.SUPPRESS)
    p_tune = sub.add_parser("tune", help="fit the compaction idle threshold to your own history (local)")
    p_tune.add_argument("--dry-run", action="store_true", help="show the choice, change nothing")
    p_tune.add_argument("--force", action="store_true", help="run even before enough real token data")

    p_bench = sub.add_parser("benchmark", help="what TokenSaver would have saved on this machine's own history "
                                                "(reads transcripts, changes nothing; free token counts)")
    p_bench.add_argument("--step", type=int, default=8, help="count every Nth request (default 8)")
    p_bench.add_argument("--since", metavar="YYYY-MM-DD", help="only sessions ending on or after this date")
    p_bench.add_argument("--yes", action="store_true", help="start without asking")
    p_bench.add_argument("--out", type=Path, help="where to write the shareable result (numbers only)")

    p_cfg = sub.add_parser("config", help="show or change settings (~/.tokensaver/config.json)")
    p_cfg.add_argument("action", nargs="?", choices=("list", "get", "set", "unset"), default="list")
    p_cfg.add_argument("name", nargs="?")
    p_cfg.add_argument("value", nargs="?")

    argv = list(sys.argv[1:] if argv is None else argv)
    # Any command that would stop the proxy refuses from a routed Claude session
    # (service.stopping_proxy_is_unsafe); this says "I am outside it, go ahead".
    if "--allow-proxy-restart" in argv:
        argv.remove("--allow-proxy-restart")
        from . import service as _sv
        _sv.ALLOW_PROXY_STOP = True
    # `tokensaver --benchmark` (a flag reads naturally for a one-off) is the benchmark command.
    if argv and argv[0] == "--benchmark":
        argv[0] = "benchmark"
    if not argv:
        ap.print_help()
        return 0
    args = ap.parse_args(argv)

    if args.cmd == "setup":
        from . import setup as st
        return st.run(dry_run=args.dry_run, yes=args.yes, route=args.route, shell_rc=args.shell_rc,
                      audit=args.audit, service=not args.no_service, with_hooks=args.with_hooks,
                      aliases=args.aliases, statusbar=args.statusbar, with_prune=args.with_prune,
                      refresh=args.refresh)
    if args.cmd == "watchdog":
        from . import watchdog
        print(watchdog.once())
        return 0
    if args.cmd == "migrate":
        from . import migrate
        return migrate.main(args.dry_run, args.yes)
    if args.cmd == "update":
        from . import update
        return update.run()
    if args.cmd == "benchmark":
        from . import benchmark
        return benchmark.run(step=args.step, since=args.since, yes=args.yes, out=args.out)
    if args.cmd == "tune":
        from . import tune as tn
        d = tn.run(apply=not args.dry_run, force=args.force)
        print(tn.render(d))
        if args.dry_run:
            print("(dry run: nothing changed)")
        return 0
    if args.cmd == "restart":
        from . import service as sv
        import time as _t
        h = sv.health()
        end = _t.time() + 600
        while h and h.get("inflight") and _t.time() < end:
            _t.sleep(1)
            h = sv.health()
        print("restarted" if sv.restart() else "restart did not come back (see ~/.tokensaver/proxy.log)")
        return 0
    if args.cmd == "config":
        from . import config as cfg
        if args.action in ("get", "set", "unset") and args.name not in cfg.SPECS:
            print(f"TokenSaver: unknown setting {args.name!r}; one of: {', '.join(cfg.SPECS)}",
                  file=sys.stderr)
            return 1
        if args.action == "set":
            if args.value is None:
                print("TokenSaver: config set NAME VALUE", file=sys.stderr)
                return 1
            cfg.set_value(args.name, args.value, "user", why="set by hand")
        elif args.action == "unset":
            cfg.unset(args.name, "user")
        if args.action == "get":
            value, source = cfg.resolve(args.name)
            print(f"{args.name} = {value!r}  ({source})")
            return 0
        print(f"settings ({cfg.PATH}); later sources win: default < tuned < user < env")
        for name, value, source, note in cfg.report():
            print(f"  {name:20s} {str(value):38s} {source:8s} {note}")
        if args.action in ("set", "unset"):
            print("restart the proxy for rewrite/port/upstream; the rest is picked up within a minute")
        return 0

    # Vault and settings commands need no transcripts.
    if args.cmd in ("search", "read", "vault", "install", "uninstall", "proxy"):
        if args.cmd == "proxy":
            if args.stats:
                print(proxy.stats())
                return 0
            if args.reset_fallback:
                return proxy.main(["--reset-fallback"])
            try:
                return proxy.serve(port=args.port, upstream=args.upstream, service=args.service)
            except OSError as exc:
                # Port in use or similar: one clear line, not a traceback.
                print(f"TokenSaver: cannot start the proxy on port {args.port}: {exc}",
                      file=sys.stderr)
                return 1
        if args.cmd == "vault":
            for k, v in vault.stats().items():
                print(f"  {k}: {v}")
        elif args.cmd == "search":
            try:
                hits = vault.search(args.query, limit=args.limit, kind=args.kind)
            except vault.QueryError as exc:
                print(f"TokenSaver: bad query: {exc}", file=sys.stderr)
                return 1
            if not hits:
                print("no matches")
            for ev in hits:
                flag = " [truncated by Claude Code]" if ev.truncated else ""
                print(f"  #{ev.id:<6} {ev.ts}  {ev.kind:<14} exit={ev.exit_code} "
                      f"{ev.bytes:>9,}b{flag}  {ev.summary}")
            return 0
        elif args.cmd == "read":
            got = vault.read(args.event_id)
            if not got:
                print(f"TokenSaver: no event {args.event_id}", file=sys.stderr)
                return 1
            ev, body = got
            if args.head:
                body = body[:args.head]
            elif args.tail:
                body = body[-args.tail:]
            print(f"# event {ev.id} {ev.kind} tool={ev.tool} exit={ev.exit_code} "
                  f"{ev.bytes:,} bytes {ev.ts}", file=sys.stderr)
            print(body)
        elif args.cmd == "install":
            print(install.install())
        else:
            # Everything setup did (service, aliases, its settings), plus hooks.
            from . import setup as st
            return st.uninstall(purge=True if args.purge else False if args.keep_data else None, yes=args.yes)
        return 0

    if not args.root.exists():
        print(f"TokenSaver: no transcript directory at {args.root}", file=sys.stderr)
        return 1
    sess = load(root=args.root, since=args.since, min_requests=args.min_requests)
    if not sess:
        print("TokenSaver: no sessions matched", file=sys.stderr)
        return 1

    if args.cmd == "report":
        print(report.render(sess, reprice_as=args.reprice_as))
    elif args.cmd == "doctor":
        print(doctor.render(sess))
    elif args.cmd == "sessions":
        print(sessions.render(sess, limit=args.limit))
    elif args.cmd == "simulate":
        print(simulate.render(sess, args.thresholds, reprice_as=args.reprice_as,
                              post_tokens=args.post_tokens, rebuild_as=args.rebuild_as))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
