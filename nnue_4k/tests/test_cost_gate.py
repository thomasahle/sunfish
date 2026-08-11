"""Unit tests for the Oracle cost tripwire's state machine (cost_gate.py).

The OCI wiring is a thin shell; everything decision-shaped lives in
CostGate.tick(now, cost_or_error) and is exercised here with synthetic
observations.  Each test pins one of the recorded design rules.
"""
import datetime as dt
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "cost_gate", ROOT / "nnue_4k" / "lichess" / "cost_gate.py")
cg = importlib.util.module_from_spec(spec)
sys.modules["cost_gate"] = cg
spec.loader.exec_module(cg)


def ts(y, m, d, h=12):
    return dt.datetime(y, m, d, h, tzinfo=dt.timezone.utc).timestamp()


def test_zero_cost_never_trips():
    g = cg.CostGate(threshold=0.0)
    for day in range(1, 28):
        assert g.tick(ts(2026, 8, day), 0.0) is False
    assert g.state()["tripped"] is False


def test_nonzero_cost_trips_and_stays_tripped_within_month():
    g = cg.CostGate(threshold=0.0)
    assert g.tick(ts(2026, 8, 10), 0.0) is False
    assert g.tick(ts(2026, 8, 11), 0.03) is True
    # a later lower reading does NOT clear it (amendment lag > refund)
    assert g.tick(ts(2026, 8, 12), 0.0) is True
    assert "0.03" in g.state()["reason"]


def test_threshold_is_strictly_greater():
    g = cg.CostGate(threshold=0.05)
    assert g.tick(ts(2026, 8, 10), 0.05) is False
    assert g.tick(ts(2026, 8, 11), 0.06) is True


def test_api_errors_do_not_trip_until_48h():
    g = cg.CostGate(threshold=0.0)
    t0 = ts(2026, 8, 10)
    assert g.tick(t0, ConnectionError("api down")) is False
    assert g.tick(t0 + 24 * 3600, ConnectionError("api down")) is False
    # one success resets the error clock
    assert g.tick(t0 + 30 * 3600, 0.0) is False
    assert g.tick(t0 + 40 * 3600, ConnectionError("down again")) is False
    assert g.tick(t0 + 40 * 3600 + 47 * 3600, ConnectionError("x")) is False
    assert g.tick(t0 + 40 * 3600 + 49 * 3600, ConnectionError("x")) is True
    assert "48h" in g.state()["reason"]


def test_month_rollover_clears_only_with_clean_reading():
    g = cg.CostGate(threshold=0.0)
    assert g.tick(ts(2026, 8, 20), 1.25) is True
    # new month, still showing cost: stays tripped
    assert g.tick(ts(2026, 9, 1), 0.40) is True
    # new month reads clean: auto-clears
    g2 = cg.CostGate(threshold=0.0)
    g2.tick(ts(2026, 8, 20), 1.25)
    assert g2.tick(ts(2026, 9, 1), 0.0) is False
    assert g2.state()["tripped"] is False


def test_state_round_trip():
    g = cg.CostGate(threshold=0.0)
    g.tick(ts(2026, 8, 20), 0.5)
    g2 = cg.CostGate(threshold=0.0, state=g.state())
    assert g2.tripped and g2.tripped_month == "2026-08"
    # restored gate still honors rollover-clear
    assert g2.tick(ts(2026, 9, 2), 0.0) is False


def test_usage_window_is_day_granular_and_covers_now():
    """The OCI Usage API 400s on sub-day timestamps (live deploy finding):
    both bounds must be midnight-aligned, and the end must round UP past
    `now` so today's accruals are inside the window."""
    for now in (dt.datetime(2026, 8, 11, 9, 41, 23, 456789,
                            tzinfo=dt.timezone.utc),
                dt.datetime(2026, 8, 1, 0, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2026, 12, 31, 23, 59, 59,
                            tzinfo=dt.timezone.utc)):
        start, end = cg._usage_window(now)
        for b in (start, end):
            assert (b.hour, b.minute, b.second, b.microsecond) == (0, 0, 0, 0), \
                f"sub-day timestamp {b} would 400 the Usage API"
        assert start == now.replace(day=1, hour=0, minute=0, second=0,
                                    microsecond=0)
        assert end > now, "end must round UP or today's cost is invisible"
        assert end - now <= dt.timedelta(days=1)


def test_service_unit_does_not_shadow_the_flag_file():
    """systemd's RuntimeDirectory= creates /run/<name> as a DIRECTORY; a
    name equal to the flag file's basename makes the daemon crashloop on
    IsADirectoryError (live deploy finding). The unit must not declare
    a RuntimeDirectory that collides with any /run path the gate writes."""
    unit = (ROOT / "nnue_4k" / "lichess" / "sunfish-cost-gate.service").read_text()
    flag_names = {"sunfish-costgate"}   # cost_gate.py --flag default basename
    for line in unit.splitlines():
        line = line.strip()
        if line.startswith("RuntimeDirectory="):
            declared = set(line.split("=", 1)[1].split())
            clash = declared & flag_names
            assert not clash, (
                f"RuntimeDirectory {clash} would shadow the flag file "
                "/run/sunfish-costgate with a directory")
