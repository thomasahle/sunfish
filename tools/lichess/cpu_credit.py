#!/usr/bin/env python3
"""CPU-credit estimator and challenge gate for burstable cloud VMs.

Background
----------
The sunfish bot runs on a GCE ``e2-micro``: two vCPUs are visible to the
guest, but the instance is only entitled to a *baseline* of 0.25 vCPU
sustained.  Above that it burst-runs on the host's spare capacity.  With
pondering enabled the engine computes on the opponent's turn too, so a game
draws roughly 0.75 vCPU against a 0.25 vCPU/s entitlement -- if a burst
budget really is finite, sustained play must eventually exhaust it.

MEASURED vs MODELLED
--------------------
Be precise about which is which; the gate is only as good as the model.

MEASURED (read straight from the kernel, no assumptions):
  * ``usage`` -- busy CPU-seconds per wall-clock second, from ``/proc/stat``
    (user+nice+system+irq+softirq), differenced against a *wall-clock*
    interval.  NB: we deliberately do **not** divide by the total tick delta.
    On this tickless kernel (CONFIG_NO_HZ_IDLE) the idle CPU stops taking
    ticks, so over short windows the columns sum to well under
    ``ncpu * HZ * dt`` (measured: ~105-115 tick/s against an expected 200 on
    a 2-vCPU box).  Normalising by the tick total would inflate usage by
    nearly 2x.
  * ``steal`` -- ticks the hypervisor ran someone else while this vCPU was
    runnable.  Nonzero steal is the only *direct* evidence of throttling the
    guest can see.

MODELLED (assumptions, not facts -- tune via the constants below):
  * ``BASELINE_VCPU`` -- the 0.25 vCPU entitlement.  This one is actually
    corroborated: the GCE metric ``instance/cpu/reserved_cores`` reads 0.25
    for this instance, and ``instance/cpu/utilization`` is expressed in units
    of it (verified against /proc/stat: a 0.58-unit reading coincided with a
    measured 0.145 vCPU, i.e. 0.58 * 0.25).
  * ``BUDGET_VCPU_SECONDS`` -- the size of the burst bucket.  **GCE publishes
    no such number and exposes no credit metric.**  Unlike AWS (which has
    ``CPUCreditBalance``/``CPUSurplusCreditBalance``), the only CPU metrics
    Cloud Monitoring exposes for a GCE instance are ``guest_visible_vcpus``,
    ``reserved_cores``, ``scheduler_wait_time``, ``usage_time`` and
    ``utilization`` -- none of them a credit balance.  (``scheduler_wait_time``
    reads a flat 0.0 on this instance, so it is not a usable throttle signal
    either.)  The bucket size here is therefore a pure guess, chosen well
    above anything ever observed in production; see the note on thresholds.
  * The linear ``deficit += (usage - baseline) * dt`` integration itself is a
    model of the bucket's dynamics, borrowed from the AWS T-series credit
    model.  It may not match GCE's actual (undocumented) scheduler.

Threshold sizing
----------------
Over the only 3.75 days of monitoring data this instance has (it booted
2026-08-05T15:06Z), the worst integrated deficit this model ever reaches is
**16.5 vCPU-seconds**, and mean usage is 0.031 vCPU -- 12.6% of the 0.25
baseline.  The instance spends essentially all of its time *accruing*.  The
defaults below therefore sit two orders of magnitude above anything observed,
which makes this gate a genuine safety net rather than a duty-cycle limiter.
If you tighten them, re-measure first: a gate that closes on a bot that is not
actually credit-starved just takes it offline for nothing.

Because the bucket size is unverified, the *steal-based* signal is the one to
trust.  Sustained nonzero steal is measured ground truth that the hypervisor
is holding the guest back; the integrated deficit is a leading indicator that
may simply be wrong.

Usage
-----
    cpu_credit.py --daemon        # sample forever, maintain the flag file
    cpu_credit.py --status        # print one JSON snapshot and exit

Tests live in ``tests/test_cpu_credit.py`` and drive the estimator against a
synthetic /proc/stat, so drain and recovery can be simulated without waiting
for real time to pass.

The daemon maintains a flag file (default ``/run/sunfish-throttled``).  Its
*presence* means "decline new challenges".  lichess-bot consults it through
the supported ``extra_game_handlers.is_supported_extra`` hook, so no patching
of lichess-bot is required, and only *new* challenges are affected -- a game
already in progress is never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

# --- Tunables -------------------------------------------------------------
# vCPU the instance is entitled to sustain. MEASURED via GCE reserved_cores.
BASELINE_VCPU = 0.25

# Size of the modelled burst bucket, in vCPU-seconds. MODELLED, unverified:
# GCE documents no bucket size and exposes no credit metric. 3600 vCPU-s is
# ~4 hours of a full-speed pondering game's surplus draw. Observed worst case
# in production is 16.5 vCPU-s, so this is deliberately generous.
BUDGET_VCPU_SECONDS = 3600.0

# Gate hysteresis, as a fraction of the budget. Close the gate when the
# modelled deficit crosses CLOSE_AT, reopen only once it recovers past
# OPEN_AT. Hysteresis stops the bot flapping between online and offline.
CLOSE_AT = 0.75
OPEN_AT = 0.40

# A game must be affordable *before* we accept it: require this much headroom
# on top of OPEN_AT. MODELLED from the ponder drain (0.75 vCPU) times a
# typical game length. 300 vCPU-s covers a ~7 minute pondering game.
GAME_RESERVE_VCPU_SECONDS = 300.0

# Steal rate (in vCPU) above which we call the instance "throttled now", and
# how long it must persist. MEASURED signal, modelled threshold. Idle noise on
# this host is a single tick every few minutes (~0.002 vCPU in a 5 s window),
# so 0.02 sustained for 30 s is comfortably above the noise floor.
STEAL_VCPU_THRESHOLD = 0.02
STEAL_SUSTAIN_SECONDS = 30.0

DEFAULT_FLAG = "/run/sunfish-throttled"
DEFAULT_STATE = "/var/lib/sunfish/credit-state.json"
DEFAULT_INTERVAL = 5.0

# Ignore absurd sample gaps (suspend, clock jump, daemon restart): integrating
# across them would invent a huge deficit or a huge recovery from nothing.
MAX_SAMPLE_GAP = 300.0

PROC_STAT = "/proc/stat"


def _clock_ticks() -> float:
    """USER_HZ, i.e. the unit of the /proc/stat columns."""
    try:
        return float(os.sysconf("SC_CLK_TCK")) or 100.0
    except (ValueError, OSError, AttributeError):
        return 100.0


class CpuSampler:
    """Reads cumulative busy/steal CPU-seconds from /proc/stat.

    Purely MEASURED: no modelling happens here.
    """

    def __init__(self, path: str = PROC_STAT) -> None:
        self.path = path
        self.hz = _clock_ticks()

    def read(self) -> tuple[float, float]:
        """Return (busy_cpu_seconds, steal_cpu_seconds), both cumulative."""
        with open(self.path) as fh:
            for line in fh:
                if line.startswith("cpu "):
                    f = [int(x) for x in line.split()[1:]]
                    break
            else:
                raise RuntimeError(f"no aggregate 'cpu' line in {self.path}")
        # user nice system idle iowait irq softirq steal guest guest_nice
        f += [0] * (10 - len(f))
        user, nice, system, _idle, _iowait, irq, softirq, steal = f[:8]
        # 'guest' time is already included in 'user' by the kernel, and
        # iowait is not CPU work, so neither belongs in the busy total.
        busy = (user + nice + system + irq + softirq) / self.hz
        return busy, steal / self.hz


class CreditEstimator:
    """Integrates (usage - baseline) into a modelled credit deficit.

    ``deficit`` is in vCPU-seconds and clamped to [0, BUDGET_VCPU_SECONDS]:
    0 means "bucket full, nothing owed", BUDGET means "modelled bucket empty".
    """

    def __init__(
        self,
        sampler: CpuSampler | None = None,
        baseline: float = BASELINE_VCPU,
        budget: float = BUDGET_VCPU_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self.sampler = sampler if sampler is not None else CpuSampler()
        self.baseline = baseline
        self.budget = budget
        self.clock = clock

        self.deficit = 0.0
        self.usage = 0.0          # last measured vCPU usage
        self.steal = 0.0          # last measured vCPU steal
        self.steal_since: float | None = None  # when steal first went high
        self.gate_open = True     # start optimistic; fail open
        self._prev: tuple[float, float, float] | None = None  # busy, steal, t

    # -- persistence -------------------------------------------------------
    def load(self, path: str) -> None:
        """Restore the modelled deficit across a restart. Best effort."""
        try:
            with open(path) as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return
        self.deficit = max(0.0, min(self.budget, float(st.get("deficit", 0.0))))
        self.gate_open = bool(st.get("gate_open", True))
        # Credits accrue while we are not running, but we cannot know the
        # machine was idle, so only credit the conservative case: assume the
        # box idled at the observed long-run mean rather than at zero.
        wall = st.get("wall_time")
        if isinstance(wall, (int, float)):
            elapsed = max(0.0, time.time() - float(wall))
            self.deficit = max(0.0, self.deficit - self.baseline * min(elapsed, 3600.0))

    def save(self, path: str) -> None:
        """Atomically persist state so a restart does not forget the deficit."""
        st = {
            "deficit": round(self.deficit, 3),
            "gate_open": self.gate_open,
            "usage": round(self.usage, 4),
            "steal": round(self.steal, 4),
            "wall_time": time.time(),
        }
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".credit-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(st, fh)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- the model ---------------------------------------------------------
    @property
    def throttled_now(self) -> bool:
        """MEASURED: steal has been above the noise floor for long enough."""
        if self.steal_since is None:
            return False
        return (self.clock() - self.steal_since) >= STEAL_SUSTAIN_SECONDS

    @property
    def headroom(self) -> float:
        """Modelled vCPU-seconds of burst left before the gate closes."""
        return max(0.0, self.budget * CLOSE_AT - self.deficit)

    def sample(self) -> dict:
        """Take one sample and update the model. Returns a status dict."""
        busy, steal = self.sampler.read()
        now = self.clock()

        if self._prev is not None:
            pbusy, psteal, ptime = self._prev
            dt = now - ptime
            # Guard against clock jumps, counter resets (reboot) and long gaps.
            if 0 < dt <= MAX_SAMPLE_GAP and busy >= pbusy and steal >= psteal:
                self.usage = (busy - pbusy) / dt
                self.steal = (steal - psteal) / dt
                # The core integration: surplus draw depletes, idle refills.
                self.deficit += (self.usage - self.baseline) * dt
                self.deficit = max(0.0, min(self.budget, self.deficit))
                # Track how long steal has been sustained.
                if self.steal >= STEAL_VCPU_THRESHOLD:
                    if self.steal_since is None:
                        self.steal_since = now
                else:
                    self.steal_since = None
            else:
                # Unusable interval: keep the deficit, drop the rate estimates.
                self.usage = self.steal = 0.0
                self.steal_since = None

        self._prev = (busy, steal, now)
        self._update_gate()
        return self.status()

    def _update_gate(self) -> None:
        """Apply hysteresis. Only ever changes ``gate_open``."""
        if self.gate_open:
            # Close if the model says we are nearly out, or if the kernel says
            # we are being throttled right now.
            if self.deficit >= self.budget * CLOSE_AT or self.throttled_now:
                self.gate_open = False
        else:
            # Reopen only once well recovered, not throttled, and with enough
            # headroom left to actually finish a game.
            recovered = self.deficit <= self.budget * OPEN_AT
            affordable = self.headroom >= GAME_RESERVE_VCPU_SECONDS
            if recovered and affordable and not self.throttled_now:
                self.gate_open = True

    def status(self) -> dict:
        return {
            "gate_open": self.gate_open,
            "deficit_vcpu_s": round(self.deficit, 1),
            "budget_vcpu_s": self.budget,
            "fraction_used": round(self.deficit / self.budget, 4) if self.budget else 0.0,
            "headroom_vcpu_s": round(self.headroom, 1),
            "usage_vcpu": round(self.usage, 4),
            "steal_vcpu": round(self.steal, 4),
            "throttled_now": self.throttled_now,
        }


def set_flag(path: str, present: bool, reason: str = "") -> None:
    """Create or remove the gate flag file.

    Presence means "decline new challenges". Written atomically so the bot
    never reads a half-written file.
    """
    if present:
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".throttled-", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            fh.write(reason or "cpu credit low")
        os.replace(tmp, path)
    else:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def run_daemon(flag: str, state: str, interval: float, verbose: bool = False) -> None:
    est = CreditEstimator()
    est.load(state)
    est.sample()  # prime the differencer; first sample yields no rate
    last_save = 0.0
    while True:
        time.sleep(interval)
        try:
            st = est.sample()
        except (OSError, RuntimeError) as exc:
            # Never let a read error take the bot offline: fail open.
            print(f"cpu_credit: sample failed: {exc}", file=sys.stderr, flush=True)
            set_flag(flag, False)
            continue
        reason = (
            "throttled: sustained CPU steal" if est.throttled_now
            else f"cpu credit low ({st['fraction_used']:.0%} of modelled budget used)"
        )
        set_flag(flag, not est.gate_open, reason)
        if verbose:
            print(json.dumps(st), flush=True)
        now = time.monotonic()
        if now - last_save >= 30.0:
            try:
                est.save(state)
            except OSError as exc:
                print(f"cpu_credit: save failed: {exc}", file=sys.stderr, flush=True)
            last_save = now


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--daemon", action="store_true", help="run continuously")
    p.add_argument("--status", action="store_true", help="print one snapshot")
    p.add_argument("--flag", default=DEFAULT_FLAG)
    p.add_argument("--state", default=DEFAULT_STATE)
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.daemon:
        run_daemon(args.flag, args.state, args.interval, args.verbose)
        return 0
    # Default: --status. Two samples a second apart so the rates are real.
    est = CreditEstimator()
    est.load(args.state)
    est.sample()
    time.sleep(1.0)
    print(json.dumps(est.sample(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
