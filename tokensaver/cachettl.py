"""How long a request's prompt cache lives: 5 minutes or 1 hour.

Claude Code picks it per request (read from its 2.1.287 source):
1. `FORCE_PROMPT_CACHING_5M` → 5m;
2. `CLAUDE_CODE_PROMPT_CACHE_TTL` (main thread) / `CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL`;
3. the settings `promptCacheTtl` / `subagentPromptCacheTtl`;
4. a subagent's own frontmatter TTL;
5. `ENABLE_PROMPT_CACHING_1H` (`_BEDROCK` on Bedrock) → 1h;
6. otherwise: no Claude subscription login (API key, Console, gateway, Bedrock, Vertex) → 5m;
   a subscription in extra usage (overage) → 5m; a subscription otherwise → 1h for the main
   thread only, 5m for subagents and side requests.

So it differs per user, per request class and even mid-session (a subscriber going into
overage). Nothing here guesses which rule applied: the request body says it exactly (the
`cache_control` markers' `ttl`; no ttl means 5m), and so does the usage of any request that
wrote to the cache (`ephemeral_5m_input_tokens` vs `ephemeral_1h_input_tokens`).
"""

from __future__ import annotations

SECONDS = {"5m": 300, "1h": 3600}
DEFAULT = "1h"        # unknown: TokenSaver's behaviour before the TTL was read


def _marker(block) -> str | None:
    cc = block.get("cache_control") if isinstance(block, dict) else None
    if not isinstance(cc, dict):
        return None
    return "1h" if cc.get("ttl") == "1h" else "5m"


def of_request(body: dict) -> str | None:
    """The TTL of the last cache marker in `messages` (the entry the next request reads),
    else of the system markers; None if the request caches nothing."""
    if not isinstance(body, dict):
        return None
    for m in reversed(body.get("messages") or []):
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for b in reversed(content):
                t = _marker(b)
                if t:
                    return t
    system = body.get("system")
    if isinstance(system, list):
        for b in reversed(system):
            t = _marker(b)
            if t:
                return t
    return None


def of_usage(r) -> str | None:
    """From a transcripts.Request: the TTL most of its cache write went to; None if it
    wrote nothing or the split is not reported."""
    w5, w1 = int(getattr(r, "write_5m", 0) or 0), int(getattr(r, "write_1h", 0) or 0)
    if not (w5 or w1):
        return None
    return "1h" if w1 >= w5 else "5m"


def per_request(reqs) -> list[str]:
    """Each request's TTL from its usage: carried forward over requests that wrote nothing,
    the session's first ones take the first TTL seen; none reported at all = DEFAULT."""
    seen = [of_usage(r) for r in reqs]
    first = next((t for t in seen if t), DEFAULT)
    out, cur = [], first
    for t in seen:
        cur = t or cur
        out.append(cur)
    return out


def seconds(ttl: str | None) -> int:
    return SECONDS.get(ttl or DEFAULT, SECONDS[DEFAULT])


def write_price(p, ttl: str | None) -> float:
    return p.write_5m if (ttl or DEFAULT) == "5m" else p.write_1h
