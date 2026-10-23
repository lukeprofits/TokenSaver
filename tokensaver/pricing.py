"""Model prices, USD per million tokens, and safe resolution of model names.

Two rules govern this file:

1. **Never guess silently.** A price that is assumed rather than verified is
   returned with `assumed=True` and a reason, and every caller surfaces it.
2. **Err toward overstating cost.** When a model is unknown but clearly belongs
   to a known family, use the most expensive verified row in that family.
   Understating cost would make every saving in this tool look better than it
   is, which is the one direction of error that matters here.

Verified against platform.claude.com/docs/en/about-claude/pricing on the date
in VERIFIED. A stale row invalidates every number the tool prints, so `doctor`
warns once that date goes cold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

VERIFIED = "2026-10-09"
STALE_AFTER_DAYS = 60


@dataclass(frozen=True)
class Price:
    inp: float       # uncached input
    out: float       # output
    write_5m: float  # 5-minute cache write
    write_1h: float  # 1-hour cache write
    read: float      # cache hit / refresh


@dataclass(frozen=True)
class Resolution:
    """The outcome of looking up a model. `price is None` means do not bill it."""
    price: Price | None
    model_id: str
    assumed: bool = False
    reason: str = ""

    @property
    def billable(self) -> bool:
        return self.price is not None and self.price.read > 0


# Exact IDs as they appear in transcripts, newest first within each family.
PRICES: dict[str, Price] = {
    "claude-fable-5-1":  Price(10.0, 50.0, 12.50, 20.0, 0.25),
    "claude-fable-5":    Price(10.0, 50.0, 12.50, 20.0, 1.00),
    "claude-opus-5-5":   Price(4.0,  20.0, 5.00,  8.0,  0.20),
    "claude-opus-5":     Price(5.0,  25.0, 6.25,  10.0, 0.50),
    "claude-opus-4-8":   Price(5.0,  25.0, 6.25,  10.0, 0.50),
    "claude-sonnet-5-5": Price(2.0,  10.0, 2.50,  4.0,  0.10),
    "claude-sonnet-5":   Price(2.0,  10.0, 2.50,  4.0,  0.20),
    "claude-sonnet-4-6": Price(3.0,  15.0, 3.75,  6.0,  0.30),
    "claude-haiku-4-5":  Price(1.0,  5.0,  1.25,  2.0,  0.10),
}

# Bare family names accepted in settings.json ("model": "opus[1m]").
#
# These resolve to the *most expensive* verified row in the family, not the
# newest, for the same reason unknown models do: which model an alias actually
# names is Claude Code's decision, not ours, and it depends on the installed
# version. Measured: on v2.1.274 `opus` resolves to claude-opus-5, while
# `claude-opus-5-5` is rejected outright as unsupported - so assuming the
# newest row would have priced this machine's traffic at rates it is not
# getting, understating the bill. `doctor` cross-checks the assumption against
# the models the proxy actually saw on the wire.
FAMILY_ALIASES = ("opus", "sonnet", "haiku", "fable")

# Runtimes billed by someone other than Anthropic.
LOCAL_PREFIXES = ("qwen", "llama", "mistral", "gemma", "deepseek", "phi", "ollama/")


def _family(model_id: str) -> str | None:
    for fam in FAMILY_ALIASES:
        if f"-{fam}-" in f"-{model_id}-":
            return fam
    return None


def _priciest_in_family(fam: str) -> tuple[str, Price]:
    """Most expensive verified row in a family, by input price then output."""
    rows = [(mid, p) for mid, p in PRICES.items() if _family(mid) == fam]
    return max(rows, key=lambda kv: (kv[1].inp, kv[1].out))


def resolve(model: str) -> Resolution:
    """Resolve any model string - concrete id, settings alias, or unknown."""
    if not model:
        return Resolution(None, "", reason="no model recorded")

    raw = model.strip()
    # Claude Code appends variant suffixes, e.g. "opus[1m]".
    base = raw.split("[")[0].strip()

    if any(base.lower().startswith(p) for p in LOCAL_PREFIXES):
        return Resolution(Price(0, 0, 0, 0, 0), base, reason="local runtime; not billed by Anthropic")

    if base in PRICES:
        return Resolution(PRICES[base], base)

    # Bare family name from settings.json, e.g. "opus".
    if base in FAMILY_ALIASES:
        mid, price = _priciest_in_family(base)
        return Resolution(price, mid, assumed=True,
                          reason=f"alias {base!r} priced at {mid}, the most expensive "
                                 f"verified {base} row, because Claude Code decides what "
                                 f"the alias names and the answer depends on its version; "
                                 f"`doctor` compares this against what the proxy saw")

    # Dated snapshot, e.g. claude-haiku-4-5-20251001.
    head, _, tail = base.rpartition("-")
    if tail.isdigit() and head in PRICES:
        return Resolution(PRICES[head], head)

    # Unknown but recognisably in a known family: overstate rather than skip.
    fam = _family(base)
    if fam:
        mid, price = _priciest_in_family(fam)
        return Resolution(price, base, assumed=True,
                          reason=f"unknown {fam} model; priced at {mid} rates, the most "
                                 f"expensive verified row in that family")

    return Resolution(None, base, reason="no verified rate; excluded from cost")


def price_for(model: str) -> Price | None:
    """Back-compat shorthand. Prefer resolve() so assumptions stay visible."""
    return resolve(model).price


def stale_days() -> int:
    y, m, d = (int(x) for x in VERIFIED.split("-"))
    return (date.today() - date(y, m, d)).days
