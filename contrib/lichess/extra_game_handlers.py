"""lichess-bot hook: decline new challenges while the CPU-credit gate is shut.

Copied to /opt/lichess-bot/extra_game_handlers.py by setup.sh, replacing the
stub that ships with lichess-bot.

Why this file and not a patch
-----------------------------
``is_supported_extra`` is lichess-bot's documented extension point for
"decide whether to accept a challenge" (lib/model.py calls it as the last
term of ``Challenge.is_supported``).  Using it means:

  * no patching of lichess-bot, so ``git pull`` in /opt/lichess-bot keeps
    working and upgrades cannot silently drop the gate;
  * it runs only on the *incoming challenge* path (``handle_challenge``), so
    a game already in progress is never interrupted -- the hard requirement;
  * a declined challenge gets a real lichess ``decline`` API call with a
    reason, not a silent timeout.

Two properties of the caller drive the implementation:

1. ``Challenge.is_supported`` wraps everything in ``try/except`` and returns
   *decline* on any exception.  A bug in here would therefore take the bot
   permanently offline, which is far worse than the throttling it guards
   against.  So every path below is wrapped and **fails open**.
2. It runs inside lichess-bot's main event loop -- the same loop that must
   stay responsive to read the opponent's moves.  So this does one ``stat``
   of a tmpfs path and nothing else: no network, no subprocess, no lock.

The decline reason lichess shows is ``generic`` ("Challenge declined"),
because lib/model.py hardcodes that for this hook.  See README.md for an
optional one-line local patch that upgrades it to ``later`` ("This is not a
good time for me, please ask again later"), which is what we actually mean.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

# Presence of this file means "decline new challenges". Maintained by
# cpu_credit.py --daemon (sunfish-credit-gate.service). Kept on /run, which is
# a tmpfs: the check costs a stat of a page already in memory, and the flag
# cannot survive a reboot as a stale lockout.
FLAG_PATH = os.environ.get("SUNFISH_THROTTLE_FLAG", "/run/sunfish-throttled")

# If the estimator daemon dies, its flag file freezes in whatever state it was
# last in. A stale "closed" flag would keep the bot offline forever, so treat
# a flag older than this as expired and ignore it -- fail open, again.
FLAG_MAX_AGE_SECONDS = 300.0

_last_logged = 0.0


def _log_throttled(reason: str) -> None:
    """Log a decline, but at most once a minute so the journal stays readable."""
    global _last_logged
    now = time.monotonic()
    if now - _last_logged >= 60.0:
        _last_logged = now
        logger.info("Declining challenges: %s", reason)


def gate_is_open(flag_path: str | None = None,
                 max_age: float | None = None) -> bool:
    """True if new games may be accepted.

    Fails open on every error: no flag, unreadable flag, stale flag, or any
    unexpected exception all mean "accept". Being occasionally throttled is a
    much smaller problem than being permanently offline.

    The defaults are resolved from the module globals at *call* time, not
    bound at import time, so operators (and tests) can retarget FLAG_PATH.
    """
    flag_path = FLAG_PATH if flag_path is None else flag_path
    max_age = FLAG_MAX_AGE_SECONDS if max_age is None else max_age
    try:
        st = os.stat(flag_path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.warning("CPU-credit flag %s unreadable (%s); accepting", flag_path, exc)
        return True

    age = time.time() - st.st_mtime
    if age > max_age:
        logger.warning("CPU-credit flag %s is stale (%.0fs old); accepting -- "
                       "is sunfish-credit-gate.service running?", flag_path, age)
        return True

    try:
        with open(flag_path) as fh:
            reason = fh.read(200).strip()
    except OSError:
        reason = "cpu credit low"
    _log_throttled(reason or "cpu credit low")
    return False


def game_specific_options(game):  # noqa: ARG001
    """Per-game engine options. We use the ones in config.yml unchanged."""
    return {}


def is_supported_extra(challenge) -> bool:  # noqa: ARG001
    """Accept a challenge only while the CPU-credit gate is open."""
    try:
        return gate_is_open()
    except Exception:  # noqa: BLE001 -- must never propagate; see module docstring
        logger.exception("CPU-credit gate check failed; accepting anyway")
        return True
