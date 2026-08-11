"""Tests for the CPU-credit estimator and the lichess-bot challenge gate.

Everything here is offline and synthetic:

* the estimator is driven by a fake ``/proc/stat`` and a fake clock, so drain
  and recovery happen in microseconds instead of hours;
* the lichess-bot hook is exercised against a stub challenge object, so no
  lichess connection (and no lichess-bot checkout) is needed.

The numbers the tests assert against come from the production measurements
recorded in tools/lichess/cpu_credit.py.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

CONTRIB = Path(__file__).resolve().parent.parent / "tools" / "lichess"


def _load(name):
    """Import a module from tools/lichess/ by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location(name, CONTRIB / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cpu_credit = _load("cpu_credit")
extra_game_handlers = _load("extra_game_handlers")


class FakeClock:
    """A monotonic clock we advance by hand."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeSampler:
    """Stands in for CpuSampler, accumulating busy/steal at a settable rate."""

    def __init__(self):
        self.busy = 0.0
        self.steal = 0.0
        self.usage_vcpu = 0.0   # vCPU of busy time accrued per wall second
        self.steal_vcpu = 0.0

    def advance(self, dt):
        self.busy += self.usage_vcpu * dt
        self.steal += self.steal_vcpu * dt

    def read(self):
        return self.busy, self.steal


def make_est(**kw):
    clock, sampler = FakeClock(), FakeSampler()
    est = cpu_credit.CreditEstimator(sampler=sampler, clock=clock, **kw)
    est.sample()  # prime the differencer
    return est, clock, sampler


def run(est, clock, sampler, seconds, step=5.0):
    """Advance the simulation, sampling every `step` seconds."""
    for _ in range(int(seconds / step)):
        clock.advance(step)
        sampler.advance(step)
        est.sample()


# --- the measured baseline ------------------------------------------------

def test_idle_box_never_closes_the_gate():
    """At the measured production idle rate the gate must never close.

    Mean usage measured over 3.75 days on the live VM is 0.031 vCPU against a
    0.25 vCPU baseline. A whole simulated day of that must leave the deficit
    at zero -- if this fails, the gate would take a healthy bot offline.
    """
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 0.031
    run(est, clock, sampler, 24 * 3600, step=30.0)
    assert est.gate_open
    assert est.deficit == 0.0


def test_production_workload_stays_far_from_the_threshold():
    """Real duty cycle: ~3 min of pondering game, then idle, repeated.

    Measured peak usage during a game is ~0.48 vCPU (1-minute mean). Even
    back-to-back games at that rate with only short gaps must stay well below
    the close threshold, matching the observed worst-case deficit of 16.5
    vCPU-seconds.
    """
    est, clock, sampler = make_est()
    for _ in range(20):
        sampler.usage_vcpu = 0.48
        run(est, clock, sampler, 180)
        sampler.usage_vcpu = 0.013     # measured deep-idle rate
        run(est, clock, sampler, 600)
    assert est.gate_open
    assert est.deficit < 100.0, f"deficit {est.deficit} unexpectedly high"


# --- drain and recovery ---------------------------------------------------

def test_sustained_overload_closes_the_gate():
    """A pondering game at 0.75 vCPU forever must eventually shut the gate."""
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 0.75
    # Surplus is 0.5 vCPU/s, so CLOSE_AT (0.75 * 3600 = 2700 vCPU-s) needs
    # 5400 s of solid play. Run well past that.
    run(est, clock, sampler, 8000)
    assert not est.gate_open
    assert est.deficit >= cpu_credit.BUDGET_VCPU_SECONDS * cpu_credit.CLOSE_AT


def test_gate_reopens_after_enough_idle():
    """Once the box idles, credits recover and the gate opens again."""
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 0.75
    run(est, clock, sampler, 8000)
    assert not est.gate_open

    sampler.usage_vcpu = 0.0          # fully idle: recovers at the baseline
    run(est, clock, sampler, 12000)
    assert est.gate_open
    assert est.deficit <= cpu_credit.BUDGET_VCPU_SECONDS * cpu_credit.OPEN_AT


def test_hysteresis_prevents_flapping():
    """Just past the close point, a whisker of recovery must not reopen."""
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 0.75
    run(est, clock, sampler, 8000)
    assert not est.gate_open

    # The deficit saturates at the full budget, so recovering below CLOSE_AT
    # takes (1 - CLOSE_AT) * budget / baseline seconds. Idle for long enough
    # to land strictly between OPEN_AT and CLOSE_AT: the gate must stay shut.
    sampler.usage_vcpu = 0.0
    run(est, clock, sampler, 5000)
    lo = cpu_credit.BUDGET_VCPU_SECONDS * cpu_credit.OPEN_AT
    hi = cpu_credit.BUDGET_VCPU_SECONDS * cpu_credit.CLOSE_AT
    assert lo < est.deficit < hi, f"deficit {est.deficit} not in the hysteresis band"
    assert not est.gate_open, "gate reopened without crossing OPEN_AT"


def test_recovery_rate_matches_the_baseline():
    """Idle recovery must run at exactly the baseline entitlement."""
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 1.0
    run(est, clock, sampler, 1000)
    before = est.deficit
    sampler.usage_vcpu = 0.0
    run(est, clock, sampler, 100)
    # 100 s of idle recovers 100 * 0.25 = 25 vCPU-seconds.
    assert before - est.deficit == pytest.approx(25.0, abs=0.5)


# --- the measured throttle signal ----------------------------------------

def test_sustained_steal_closes_the_gate_immediately():
    """Steal is ground truth: it shuts the gate regardless of the model."""
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 0.1          # well under baseline, deficit stays 0
    sampler.steal_vcpu = 0.30
    run(est, clock, sampler, 60)
    assert est.deficit == 0.0, "this test must isolate the steal path"
    assert est.throttled_now
    assert not est.gate_open


def test_brief_steal_spike_is_ignored():
    """A single tick of steal is noise, not throttling."""
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 0.1
    sampler.steal_vcpu = 0.30
    run(est, clock, sampler, 10)      # under STEAL_SUSTAIN_SECONDS
    assert not est.throttled_now
    assert est.gate_open


# --- robustness -----------------------------------------------------------

def test_counter_reset_does_not_invent_a_deficit():
    """A reboot rewinds /proc/stat; that must not read as a huge burst."""
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 0.5
    run(est, clock, sampler, 200)
    d = est.deficit
    sampler.busy = 0.0                # counters reset
    clock.advance(5.0)
    est.sample()
    assert est.deficit == d, "counter reset perturbed the deficit"


def test_long_gap_is_not_integrated():
    """A suspended/overloaded daemon must not integrate across the gap."""
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 2.0
    clock.advance(cpu_credit.MAX_SAMPLE_GAP + 100)
    sampler.advance(cpu_credit.MAX_SAMPLE_GAP + 100)
    est.sample()
    assert est.deficit == 0.0


def test_state_round_trips(tmp_path):
    """The modelled deficit survives a daemon restart."""
    est, clock, sampler = make_est()
    sampler.usage_vcpu = 1.0
    run(est, clock, sampler, 2000)
    est.gate_open = False
    path = str(tmp_path / "state.json")
    est.save(path)

    fresh = cpu_credit.CreditEstimator(sampler=FakeSampler(), clock=FakeClock())
    fresh.load(path)
    # load() credits elapsed wall time against the deficit; that is ~0 here.
    assert fresh.deficit == pytest.approx(est.deficit, abs=1.0)
    assert fresh.gate_open is False


def test_load_of_missing_or_corrupt_state_is_harmless(tmp_path):
    est = cpu_credit.CreditEstimator(sampler=FakeSampler(), clock=FakeClock())
    est.load(str(tmp_path / "nope.json"))
    assert est.deficit == 0.0 and est.gate_open

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    est.load(str(bad))
    assert est.deficit == 0.0 and est.gate_open


def test_real_proc_stat_is_readable():
    """Smoke-test the real sampler where /proc/stat exists (Linux)."""
    if not os.path.exists("/proc/stat"):
        pytest.skip("no /proc/stat on this platform")
    busy, steal = cpu_credit.CpuSampler().read()
    assert busy > 0 and steal >= 0


# Two consecutive real samples captured from the production VM
# (sunfish-lichess, us-central1-a), two seconds apart while idle.
VM_STAT_A = "cpu  780052 900 101441 61641517 94512 0 57969 6960 0 0\n"
VM_STAT_B = "cpu  780052 900 101442 61641916 94512 0 57969 6960 0 0\n"


def test_parses_a_real_vm_proc_stat(tmp_path):
    """Parse real captured production samples, not a hand-made string.

    Guards the two subtle choices in CpuSampler: iowait (column 5) is excluded
    from busy, and the reading is normalised by *wall clock* rather than by the
    tick total. Between these two samples only 1 system tick and 399 idle ticks
    elapsed over 2 s -- the columns sum to 400 tick/s against the 200 tick/s a
    2-vCPU box "should" produce, which is exactly why dividing by the tick
    total would give a nonsense answer.
    """
    p = tmp_path / "stat"
    sampler = cpu_credit.CpuSampler(path=str(p))

    p.write_text(VM_STAT_A)
    busy_a, steal_a = sampler.read()
    p.write_text(VM_STAT_B)
    busy_b, steal_b = sampler.read()

    hz = sampler.hz
    # Exactly one system tick of work, and no steal at all.
    assert (busy_b - busy_a) == pytest.approx(1.0 / hz, rel=1e-6)
    assert steal_b == steal_a
    # Over the real 2 s gap that is a vanishing 0.005 vCPU -- below the
    # STEAL/idle noise floor measured on the box.
    assert (busy_b - busy_a) / 2.0 < 0.01


def test_iowait_is_not_counted_as_busy(tmp_path):
    """iowait is waiting, not work: counting it would fake a credit drain.

    This matters on the production box specifically, which is swapping.
    """
    p = tmp_path / "stat"
    sampler = cpu_credit.CpuSampler(path=str(p))
    p.write_text("cpu  100 0 100 1000 100 0 0 0 0 0\n")
    busy_a, _ = sampler.read()
    # Add 500 ticks of pure iowait and nothing else.
    p.write_text("cpu  100 0 100 1000 600 0 0 0 0 0\n")
    busy_b, _ = sampler.read()
    assert busy_b == busy_a


def test_short_proc_stat_line_is_tolerated(tmp_path):
    """Older kernels omit the trailing guest columns; do not crash."""
    p = tmp_path / "stat"
    p.write_text("cpu  100 0 50 1000 10 0 5\n")   # no steal/guest columns
    busy, steal = cpu_credit.CpuSampler(path=str(p)).read()
    assert steal == 0.0
    assert busy == pytest.approx(155 / cpu_credit._clock_ticks())


# --- the flag file and the lichess-bot hook -------------------------------

class StubChallenge:
    """Enough of lichess-bot's model.Challenge for the hook, which ignores it."""

    def __init__(self, cid="abc123"):
        self.id = cid


def test_hook_accepts_when_no_flag(tmp_path, monkeypatch):
    flag = str(tmp_path / "throttled")
    monkeypatch.setattr(extra_game_handlers, "FLAG_PATH", flag)
    assert extra_game_handlers.is_supported_extra(StubChallenge()) is True


def test_hook_declines_while_flag_present(tmp_path, monkeypatch):
    flag = str(tmp_path / "throttled")
    monkeypatch.setattr(extra_game_handlers, "FLAG_PATH", flag)
    cpu_credit.set_flag(flag, True, "cpu credit low")
    assert extra_game_handlers.is_supported_extra(StubChallenge()) is False
    # ...and accepts again the moment the estimator clears it.
    cpu_credit.set_flag(flag, False)
    assert extra_game_handlers.is_supported_extra(StubChallenge()) is True


def test_hook_fails_open_on_stale_flag(tmp_path, monkeypatch):
    """A dead estimator must not strand the bot offline forever."""
    flag = tmp_path / "throttled"
    monkeypatch.setattr(extra_game_handlers, "FLAG_PATH", str(flag))
    cpu_credit.set_flag(str(flag), True, "cpu credit low")
    old = time.time() - extra_game_handlers.FLAG_MAX_AGE_SECONDS - 60
    os.utime(flag, (old, old))
    assert extra_game_handlers.is_supported_extra(StubChallenge()) is True


def test_hook_fails_open_on_unexpected_error(monkeypatch):
    """Any exception in the gate means accept, never decline.

    lichess-bot's Challenge.is_supported turns an exception into a decline, so
    a crashing gate would silently take the bot offline. Guard against that.
    """
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(extra_game_handlers, "gate_is_open", boom)
    assert extra_game_handlers.is_supported_extra(StubChallenge()) is True


def test_set_flag_is_idempotent(tmp_path):
    flag = str(tmp_path / "throttled")
    cpu_credit.set_flag(flag, False)          # removing a missing flag is fine
    cpu_credit.set_flag(flag, True, "one")
    cpu_credit.set_flag(flag, True, "two")    # overwrite in place
    assert Path(flag).read_text() == "two"
    cpu_credit.set_flag(flag, False)
    cpu_credit.set_flag(flag, False)
    assert not Path(flag).exists()


def test_end_to_end_gate_cycle(tmp_path, monkeypatch):
    """Drain -> flag appears -> bot declines; recover -> flag goes -> accepts."""
    flag = str(tmp_path / "throttled")
    monkeypatch.setattr(extra_game_handlers, "FLAG_PATH", flag)
    est, clock, sampler = make_est()

    def publish():
        cpu_credit.set_flag(flag, not est.gate_open, "cpu credit low")

    publish()
    assert extra_game_handlers.is_supported_extra(StubChallenge()) is True

    sampler.usage_vcpu = 0.75
    run(est, clock, sampler, 8000)
    publish()
    assert extra_game_handlers.is_supported_extra(StubChallenge()) is False

    sampler.usage_vcpu = 0.0
    run(est, clock, sampler, 12000)
    publish()
    assert extra_game_handlers.is_supported_extra(StubChallenge()) is True
