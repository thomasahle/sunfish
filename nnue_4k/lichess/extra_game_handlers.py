"""lichess-bot hook: decline new challenges while the COST gate is tripped.

Copied to /opt/lichess-bot/extra_game_handlers.py by setup.sh, replacing the
stub that ships with lichess-bot.  Same doctrine as the classic bot's
CPU-credit hook (tools/lichess/extra_game_handlers.py, where the design
rationale lives): ``is_supported_extra`` is the documented decline point, it
only runs on the incoming-challenge path (games in progress are never
interrupted), and every path below FAILS OPEN -- a bug here must throttle
the bot, never take it offline.

The flag is maintained by cost_gate.py --daemon (sunfish-cost-gate.service):
present = the Oracle account showed cost this month (or the Usage API has
been dark for 48h) -- see nnue_4k/lichess/cost_gate.py for the trip rules
and the manual clear procedure.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

FLAG_PATH = os.environ.get("SUNFISH_COST_FLAG", "/run/sunfish-costgate")

# The daemon refreshes the flag mtime every wake-up (600 s) while tripped; a
# flag much older than that means the daemon died mid-trip.  Stale flags are
# ignored -- fail open, a dead daemon must not strand a lockout.
FLAG_MAX_AGE_SECONDS = 1800.0

_last_logged = 0.0


def _log_throttled(reason: str) -> None:
    global _last_logged
    now = time.monotonic()
    if now - _last_logged >= 60.0:
        _last_logged = now
        logger.info("Declining challenges: %s", reason)


def gate_is_open(flag_path: str | None = None,
                 max_age: float | None = None) -> bool:
    """True if new games may be accepted; fails open on every error."""
    flag_path = FLAG_PATH if flag_path is None else flag_path
    max_age = FLAG_MAX_AGE_SECONDS if max_age is None else max_age
    try:
        st = os.stat(flag_path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        logger.warning("cost flag %s unreadable (%s); accepting", flag_path, exc)
        return True
    age = time.time() - st.st_mtime
    if age > max_age:
        logger.warning("cost flag %s is stale (%.0fs old); accepting -- is "
                       "sunfish-cost-gate.service running?", flag_path, age)
        return True
    try:
        with open(flag_path) as fh:
            reason = fh.read(200).strip()
    except OSError:
        reason = "cost gate tripped"
    _log_throttled(reason or "cost gate tripped")
    return False


def game_specific_options(game):  # noqa: ARG001
    return {}


def is_supported_extra(challenge) -> bool:  # noqa: ARG001
    try:
        return gate_is_open()
    except Exception:  # noqa: BLE001 -- must never propagate
        logger.exception("cost gate check failed; accepting anyway")
        return True
