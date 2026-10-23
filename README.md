# TokenSaver

**A cheaper Claude Code bill, without changing how you work.** Install it once,
keep typing `claude` like always, and it quietly trims what Claude Code resends
on every message.

```sh
git clone https://github.com/lukeprofits/TokenSaver ~/.tokensaver/src && ~/.tokensaver/src/install.sh
```

The installer shows you everything it will change and asks once. After that,
Claude Code in your terminal, VS Code or JetBrains goes through TokenSaver
automatically. It runs on any computer, no AI model needed. You'll see a small
⚡ with the % saved in Claude Code's status line.

## How much it saves

Measured on 51 real Claude Code sessions (one developer, mostly Opus):

| | plain Claude Code | with TokenSaver |
|---|---|---|
| tokens sent | 100% | **71%** (29% fewer) |
| bill | 100% | **77%** (23% cheaper) |
| bill, long sessions (200+ messages) | 100% | 73% (27% cheaper) |
| bill, medium sessions (50–199) | 100% | 93% (7% cheaper) |
| bill, short sessions (under 50) | 100% | about the same |

Long sessions with breaks in them save the most. Want your own number first?
This replays the Claude Code sessions already on your computer and changes
nothing:

```sh
git clone https://github.com/lukeprofits/TokenSaver && TokenSaver/tokensaver-cli benchmark
```

(It asks before starting: it uses Anthropic's free token counter with your
own Claude login.)

## Commands

```sh
tokensaver doctor        # is it installed and working?
tokensaver --benchmark   # what it would have saved on your past sessions
tokensaver proxy --stats # what it has saved so far
tokensaver update        # get the latest version
tokensaver uninstall     # remove it completely
```

## It won't break Claude

- It never changes what you type, and never touches what Claude is working on
  right now.
- If TokenSaver stops working, Claude Code goes straight to Anthropic, the way
  it did before you installed it. That's true even if its folder gets deleted.
- It adds its own settings next to yours, keeps a backup, and changes nothing
  else. If your company already sends Claude through its own gateway,
  TokenSaver works with it or stays out of the way.
- `tokensaver uninstall` removes everything and puts back anything it
  replaced.

---

## How it works

Every time you send Claude a message, Claude Code sends **the whole
conversation** again: every file Claude read, every command output, everything
said so far. In a long session that's most of your bill.

Anthropic gives a big discount for resending exactly what it has seen in the
last hour (this is called the cache). So you can't just trim the conversation
whenever you like: changing anything Claude already saw throws the discount
away and costs more than it saves. TokenSaver works around that with two kinds
of steps:

1. **On every message, it tidies only what's new**, before Claude sees it for
   the first time. Nothing Claude already saw changes, so the discount stays.
2. **When you take a long break**, long enough that the discount would expire
   anyway, it cleans up the older part of the conversation. Resending costs
   the same at that moment either way, so the cleanup is free and every message
   after it is smaller.

It's a small helper program on your own computer, sitting between Claude Code
and Anthropic. It doesn't use an AI model or an API key of its own.

### Step 1: on every message (instant, nothing lost)

| what it does | typical saving |
|---|---|
| **Long scripts in commands.** When Claude runs a long inline script, the history keeps a short command pointing to an exact copy saved on your computer | the biggest one: commands about half the size, ~9% of the conversation |
| **Duplicate plans.** Plan mode puts the same plan in the history twice; one copy is kept | ~3.5%, in sessions that use plan mode |
| **JSON.** Compacts JSON output without losing anything | small for most people; a lot for JSON-heavy output |
| **Clutter.** Removes color codes and trailing spaces | under 1% |
| **Repeated lines.** A line repeated many times in a row becomes "this line, repeated N times" | helps on noisy logs |

Together these save about 2% of the bill. Small, but it's on every message and
loses nothing.

### Step 2: during long breaks (the big one)

| what it does | needs | typical saving |
|---|---|---|
| **Cleanup.** The older part of the conversation is replaced with a tighter copy: everything you and Claude wrote is kept word for word; old command output and file views are cut to their first lines plus anything Claude referred to later; the most recent part stays exactly as it was | nothing, any computer | messages after it are ~39% smaller; sessions with several breaks ~30% cheaper |
| **Short names for long paths.** Long folder paths that repeat a lot get a short name, only when that actually saves tokens | nothing | 0–2%, often skipped |
| **Drop replaced pieces.** A small local AI model finds pieces that later work replaced (an old view of a file that was then edited, an attempt that was redone) | a local model, 24 GB+ memory | ~4% more |
| **Drop repeated pieces.** A larger local model finds things that were said again later | a larger local model, 48 GB+ Mac | ~3% more |
| **Trim old output.** The larger model picks which lines of old output still matter | same | ~1% more |
| **Final check.** Before any AI-made trim is used, it quizzes both versions about the old conversation; if the trimmed one loses anything, the untrimmed one is used | same | none: it's the safety step |

If Claude starts re-reading files or re-running commands more often after a
cleanup, TokenSaver turns cleanups off and tells you.

## More savings on powerful computers

Everything above "Drop replaced pieces" works on any computer with no AI
model. If your computer has the memory, you can add local AI models that run
on your own machine for free (through [ollama](https://ollama.com)) and trim a
little more during breaks:

| your computer | what you get |
|---|---|
| any computer | everything on every message, plus the break-time cleanup |
| 24 GB+ of memory | + a small model (~9 GB download) that drops replaced pieces |
| Mac with 48 GB+ | + a larger model (~22 GB) that drops repeated pieces, trims old output, and runs the final check |

That's about 0.5–1% more saving. It's opt-in, because the downloads are big:

```sh
~/.tokensaver/src/install.sh --with-prune
```

It only installs what your computer can handle, and the models pause while
you're working the machine hard.

## The ⚡ in your status line

The % is how much less your next message sends than it would without
TokenSaver.

| you see | it means |
|---|---|
| ⚡ 23% | Saving 23%, and a cleanup is ready: a longer break will save more |
| ⚡ 23% in green | Saving 23%, and that's as good as it gets for now |
| ⚡ with no number | Working, no measurement yet for this session |
| ⚡ 0% | Nothing to trim yet (a new or short session), or TokenSaver is passing this session through untouched |
| ⚡down in red | TokenSaver isn't answering. New sessions go straight to Anthropic; restart this one if it stops responding |

A low number early on is normal: savings grow as the conversation gets longer
and after your first long break.

## Limits

- macOS is tested. Linux should work but hasn't been tried on a real machine
  yet.
- Needs Python 3.9+ and Claude Code.
- Claude Code's Remote Control feature isn't available while TokenSaver is
  installed (Claude Code turns it off for custom setups like this).
- The Claude desktop app hasn't been tested.
- TokenSaver keeps a local copy of your sessions in `~/.tokensaver`, the same
  kind of material Claude Code already keeps in `~/.claude`. Nothing leaves
  your computer except your normal Claude traffic and the free token counts.

## License

MIT
