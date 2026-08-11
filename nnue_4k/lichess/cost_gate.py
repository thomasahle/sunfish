#!/usr/bin/env python3
"""Oracle cost tripwire: decline new challenges if the account accrues cost.

Why this exists
---------------
The bot runs on an Oracle A1 always-free shape: dedicated OCPUs, no
CPU-credit mechanism, so GAME LOAD CANNOT COST MONEY.  The reachable
paid-band risks are egress past the free 10 TB/month, storage growth, or a
misprovisioned shape (setup.sh statically asserts the shape; this daemon is
the belt to that braces).  Thomas's rule: if ANY cost shows up on the
account, stop taking new challenges -- finish the games in progress, stay
up, and say so loudly in the journal.

Design
------
Same split as the classic bot's CPU-credit gate (tools/lichess/): a daemon
maintains a tmpfs flag file; lichess-bot's ``is_supported_extra`` hook
(extra_game_handlers.py here) declines challenges while the flag exists and
FAILS OPEN on anything unexpected -- a bug in the gate must throttle, never
kill, the bot.

The trip logic is a small pure state machine (``CostGate.tick``), unit
tested with a fake fetcher; the OCI Usage API wiring is a thin shell around
it (instance-principal auth, no keys on disk).

Rules (each is a deliberate choice, recorded here):
  * month-to-date cost > threshold (default 0.00) trips the gate.
  * ONCE TRIPPED, STAYS TRIPPED for that calendar month, even if a later
    reading is lower (usage data amends; a cost that "goes away" is more
    likely lag than refund).  Clears automatically at month rollover if the
    new month reads clean, or manually via ``--clear``.
  * The Usage API being unreachable does NOT trip the gate -- an API outage
    must not take the bot offline (usage data lags ~24h anyway).  It logs
    loudly.  But if errors persist for 48 hours straight, trip anyway:
    flying blind for two days is no longer safe.
  * The flag file's mtime is refreshed every wake-up while tripped, so the
    handler's staleness fail-open (a daemon crash must not strand a stale
    lockout) coexists with the slow polling cadence.

usage:
  cost_gate.py --daemon [--flag PATH] [--state PATH] [--threshold USD]
  cost_gate.py --status
  cost_gate.py --clear          # manual reset after investigating
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import time

logger = logging.getLogger("cost_gate")

POLL_API_SECONDS = 3600.0        # usage data lags ~24h; hourly is generous
WAKE_SECONDS = 600.0             # flag freshness heartbeat while tripped
ERROR_FAILSAFE_SECONDS = 48 * 3600.0


class CostGate:
    """The trip logic, free of I/O.  Feed it (utc_now, fetch()) via tick."""

    def __init__(self, threshold=0.0, state=None):
        self.threshold = threshold
        s = state or {}
        self.tripped = s.get("tripped", False)
        self.tripped_month = s.get("tripped_month")   # "YYYY-MM" or None
        self.reason = s.get("reason", "")
        self.first_error_ts = s.get("first_error_ts")  # epoch or None

    def state(self):
        return {"tripped": self.tripped, "tripped_month": self.tripped_month,
                "reason": self.reason, "first_error_ts": self.first_error_ts}

    @staticmethod
    def _month(now_ts):
        return _dt.datetime.fromtimestamp(now_ts, _dt.timezone.utc).strftime("%Y-%m")

    def tick(self, now_ts, cost_or_error):
        """One observation.  cost_or_error: float month-to-date USD, or an
        Exception instance for an unreachable API.  Returns self.tripped."""
        month = self._month(now_ts)
        if isinstance(cost_or_error, Exception):
            if self.first_error_ts is None:
                self.first_error_ts = now_ts
            logger.error("Usage API unreachable (%s); NOT tripping -- "
                         "%.1fh of continuous errors (fail-safe at 48h)",
                         cost_or_error,
                         (now_ts - self.first_error_ts) / 3600.0)
            if (not self.tripped
                    and now_ts - self.first_error_ts >= ERROR_FAILSAFE_SECONDS):
                self.tripped = True
                self.tripped_month = month
                self.reason = ("usage API unreachable for 48h straight -- "
                               "flying blind, failing safe")
                logger.error("TRIPPED: %s", self.reason)
            return self.tripped

        self.first_error_ts = None
        if self.tripped:
            # month rollover with a clean reading auto-clears; anything else
            # stays tripped until --clear (see the rules in the docstring)
            if month != self.tripped_month and cost_or_error <= self.threshold:
                logger.warning("month rolled over (%s) and reads %.4f USD: "
                               "gate auto-clears", month, cost_or_error)
                self.tripped, self.tripped_month, self.reason = False, None, ""
            return self.tripped
        if cost_or_error > self.threshold:
            self.tripped = True
            self.tripped_month = month
            self.reason = ("month-to-date cost %.4f USD exceeds threshold "
                           "%.4f" % (cost_or_error, self.threshold))
            logger.error("TRIPPED: %s -- declining new challenges (games in "
                         "progress finish; clear with cost_gate.py --clear "
                         "after investigating)", self.reason)
        return self.tripped


# ---------------------------------------------------------------- OCI shell
def fetch_month_cost():
    """Month-to-date COST total for the tenancy, via instance principals.

    Any exception propagates to the caller (the state machine treats it as
    an unreachable API -- loud, non-tripping until the 48h fail-safe)."""
    import oci
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    tenancy = signer.tenancy_id
    client = oci.usage_api.UsageapiClient({}, signer=signer)
    now = _dt.datetime.now(_dt.timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    detail = oci.usage_api.models.RequestSummarizedUsagesDetails(
        tenant_id=tenancy,
        time_usage_started=start,
        time_usage_ended=now,
        granularity="MONTHLY",
        query_type="COST")
    items = client.request_summarized_usages(detail).data.items
    return float(sum(i.computed_amount or 0.0 for i in items))


def _load_state(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def daemon(flag, state_path, threshold):
    gate = CostGate(threshold, _load_state(state_path))
    last_poll = 0.0
    while True:
        now = time.time()
        if now - last_poll >= POLL_API_SECONDS:
            last_poll = now
            try:
                obs = fetch_month_cost()
                logger.info("month-to-date cost: %.4f USD (threshold %.4f)",
                            obs, threshold)
            except Exception as exc:  # noqa: BLE001 -- the machine handles it
                obs = exc
            gate.tick(now, obs)
            _save_state(state_path, gate.state())
        if gate.tripped:
            # (re)write the flag every wake-up: content for the handler's
            # log line, fresh mtime for its staleness fail-open
            with open(flag, "w") as f:
                f.write("cost gate: " + gate.reason)
        else:
            try:
                os.remove(flag)
            except FileNotFoundError:
                pass
        time.sleep(WAKE_SECONDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--flag", default="/run/sunfish-costgate")
    ap.add_argument("--state", default="/var/lib/sunfish/cost-gate.json")
    ap.add_argument("--threshold", type=float, default=0.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.clear:
        for p in (args.flag, args.state):
            try:
                os.remove(p)
                print("removed", p)
            except FileNotFoundError:
                pass
        return
    if args.status:
        print(json.dumps(_load_state(args.state) or {"tripped": False},
                         indent=2))
        print("flag present:", os.path.exists(args.flag))
        return
    if args.daemon:
        daemon(args.flag, args.state, args.threshold)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
