#!/usr/bin/env python3
"""Serial TRAINQUEUE consumer -- the always-training rule, FOR TRAININGS ONLY.

"The trainer never idles" (TRAINQUEUE.md, standing rule): the moment a run
finishes, the top queue entry starts.  This runner implements exactly that
and NOTHING else: it never launches matches, screens, gates or landings --
those stay coordinator-dispatched and play-gated.  It trains.

Queue = nnue_4k/train/queue/*.yaml, priority by filename sort (prefix with
NN_).  A finished config moves to queue/done/ with a result line appended
to queue/LOG.md.  An empty queue is a standing-rule violation and is nagged
about loudly, never silently tolerated.

Box etiquette (labeller-class, the recorded rules):
  * the training subprocess runs under `nice -n 19`
  * threads and workers are capped at 8 regardless of config
  * single writer: an atomic mkdir lock (queue/.runlock) refuses a second
    runner; stale locks are reported, never auto-broken
  * FORFEIT TRIPWIRE: live fastchess matches on the box must not pay for
    our training.  The runner baselines "loses on time" counts across the
    configured pgn globs at launch and polls while training; ANY increase
    SIGSTOPs the training, writes PAUSED_REPORT.txt beside the run and
    waits.  Removing the report file SIGCONTs (operator decision, never
    automatic) -- pause-and-report, not pause-and-guess.
  * TRIPWIRE EXEMPTION, opt-in per arena: an arena directory containing a
    file named FORFEITS_EXPECTED (one line saying why) is excluded from the
    count, and so is everything beneath it.  It exists because the guard's
    all-pgn glob counts forfeits from arenas where a forfeit is the
    INSTRUMENT, not the harm -- gauntlet-20260818's hcal cells run at
    tc=0.5+0.005, where flagging is close to inherent, and
    tmsimple-20260818/sd60.pgn registered its forfeits as data before game 1
    ("3 time forfeits, all on `simple`, as registered: data, not a void").
    With any such arena live the always-training rule could not hold: on
    2026-08-19 the trainer paused on two hcal forfeits with six arms queued.
    THE DEFAULT STAYS PARANOID -- an unmarked arena counts, exactly as
    before -- and the exemption CANNOT be made global: the marker is only
    consulted in directories strictly BELOW ~/sunfish-bench, so dropping one
    at the bench root does nothing.  Marking someone else's arena is a
    coordination act; mark your own, and say why in the file.

usage: queue_runner.py [--queue-dir DIR] [--once] [--pgn-globs G1:G2:...]
"""
import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GLOBS = os.pathsep.join((
    os.path.expanduser("~/sunfish-bench/*.pgn"),
    os.path.expanduser("~/sunfish-bench/*/*.pgn"),
))


EXEMPT_MARKER = "FORFEITS_EXPECTED"
BENCH_ROOT = os.path.abspath(os.path.expanduser("~/sunfish-bench"))


def _exempt_dir(d, cache):
    """Is `d` inside an arena that opted out of the tripwire?

    Walks up to, but NEVER reaches, BENCH_ROOT: a marker at the bench root
    is not consulted, so the exemption cannot be turned into a global off
    switch by dropping one file in the obvious place.  `cache` is per-call,
    not per-process, so an arena that marks itself mid-run is honoured at the
    next poll instead of at the next restart.
    """
    d = os.path.abspath(d)
    seen = []
    while d.startswith(BENCH_ROOT + os.sep):
        if d in cache:
            hit = cache[d]
            break
        seen.append(d)
        if os.path.exists(os.path.join(d, EXEMPT_MARKER)):
            hit = True
            break
        d = os.path.dirname(d)
    else:
        hit = False
    for x in seen:
        cache[x] = hit
    return hit


def exempt_arenas():
    """(dir, reason) for every arena currently opted out -- for the log."""
    out = []
    for m in sorted(glob.glob(os.path.join(BENCH_ROOT, "*", EXEMPT_MARKER)) +
                    glob.glob(os.path.join(BENCH_ROOT, "*", "*", EXEMPT_MARKER))):
        try:
            why = open(m).readline().strip()
        except OSError:
            why = ""
        out.append((os.path.dirname(m), why))
    return out


def forfeit_count(globs):
    """"loses on time" across the globs, skipping opted-out arenas."""
    n, cache = 0, {}
    for g in globs:
        for path in glob.glob(g):
            if _exempt_dir(os.path.dirname(path), cache):
                continue
            try:
                with open(path, "rb") as f:
                    n += f.read().count(b"loses on time")
            except OSError:
                pass
    return n


def acquire_lock(qdir):
    lock = os.path.join(qdir, ".runlock")
    try:
        os.mkdir(lock)
    except FileExistsError:
        raise SystemExit("another queue_runner holds %s -- a second runner "
                         "violates single-writer; if it is stale, a human "
                         "removes it (the lock never auto-breaks)" % lock)
    with open(os.path.join(lock, "owner"), "w") as f:
        f.write("%d %s\n" % (os.getpid(), time.strftime("%Y-%m-%dT%H:%M:%S")))
    return lock


def spawn_tail(tail, qdir):
    """Materialise the TAIL config as an ordinary queue entry, with the seed
    rotated, and return its path.

    The tail exists so an empty queue starts work instead of nagging for ten
    minutes (three gaps on 2026-08-15 cost ~100 idle minutes between them).
    It is copied rather than consumed, so `tail.yaml` survives every firing;
    the copy is a normal entry, which means it is logged to LOG.md, gets its
    own run dir from its filename, and is archived to done/ like anything
    else.  The seed advances by the number of tail runs already archived, so
    repeated firings accumulate a seed census instead of recomputing one run.
    A tail whose seed cannot be found is REFUSED, never run unrotated."""
    n = len(glob.glob(os.path.join(qdir, "done", "*_tail*.yaml")))
    with open(tail) as f:
        src = f.read()
    new, k = re.subn(r"(?<![\w])seed: (\d+)",
                     lambda m: "seed: %d" % (int(m.group(1)) + n + 1), src, count=1)
    if k != 1:
        raise SystemExit("tail config %s has no rotatable `seed: N` (found %d) -- "
                         "refusing to run it unrotated, which would repeat one run "
                         "forever instead of accumulating a census" % (tail, k))
    dst = os.path.join(qdir, "99_tail%03d.yaml" % (n + 1))
    with open(dst, "w") as f:
        f.write(new)
    return dst


def run_one(cfg_path, qdir, globs):
    name = os.path.basename(cfg_path)
    run_dir = os.path.join(_here, "runs", name.rsplit(".", 1)[0])
    os.makedirs(run_dir, exist_ok=True)
    base = forfeit_count(globs)
    ex = exempt_arenas()
    print("[queue] starting %s (forfeit baseline %d%s)"
          % (name, base,
             "".join("; EXEMPT %s: %s" % (os.path.basename(d), w or "no reason given")
                     for d, w in ex)), flush=True)
    proc = subprocess.Popen(
        ["nice", "-n", "19", sys.executable, os.path.join(_here, "train.py"),
         cfg_path, "--resume", "--out-dir", run_dir, "--threads", "8", "--workers", "8"],
        stdout=open(os.path.join(run_dir, "train.log"), "a"),
        stderr=subprocess.STDOUT)
    paused_report = os.path.join(run_dir, "PAUSED_REPORT.txt")
    paused = False
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc
        now = forfeit_count(globs)
        if now > base and not paused:
            os.kill(proc.pid, signal.SIGSTOP)
            paused = True
            with open(paused_report, "w") as f:
                f.write("TRIPWIRE: time forfeits rose %d -> %d while %s trained.\n"
                        "Training is SIGSTOPped (pid %d).  Investigate the live\n"
                        "match; delete this file to SIGCONT and continue.\n"
                        "\nCounted arenas exclude any directory holding a\n"
                        "FORFEITS_EXPECTED marker; %d are exempt right now%s.\n"
                        "If the arena that just flagged is one where forfeits\n"
                        "are the instrument rather than the harm, the fix is\n"
                        "for ITS owner to drop that marker, not to delete this\n"
                        "file on every run.\n"
                        % (base, now, name, proc.pid, len(ex),
                           "".join("\n  - %s: %s" % (os.path.basename(d), w or "no reason given")
                                   for d, w in ex)))
            print("[queue] TRIPWIRE: forfeits %d -> %d; %s PAUSED (see %s)"
                  % (base, now, name, paused_report), flush=True)
        if paused and not os.path.exists(paused_report):
            os.kill(proc.pid, signal.SIGCONT)
            paused = False
            base = forfeit_count(globs)   # re-baseline after the operator's call
            print("[queue] resumed %s (new baseline %d)" % (name, base), flush=True)
        time.sleep(120 if not paused else 30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue-dir", default=os.path.join(_here, "queue"))
    p.add_argument("--once", action="store_true", help="run one entry and exit")
    p.add_argument("--pgn-globs", default=os.environ.get("QUEUE_PGN_GLOBS", DEFAULT_GLOBS))
    a = p.parse_args()
    qdir = a.queue_dir
    done = os.path.join(qdir, "done")
    failed = os.path.join(qdir, "failed")
    os.makedirs(done, exist_ok=True)
    os.makedirs(failed, exist_ok=True)
    globs = [g for g in a.pgn_globs.split(os.pathsep) if g]
    lock = acquire_lock(qdir)
    try:
        while True:
            pending = sorted(p_ for p_ in glob.glob(os.path.join(qdir, "*.yaml")))
            if not pending:
                tail = os.path.join(os.path.dirname(os.path.abspath(qdir)), "tail.yaml")
                if os.path.exists(tail):
                    spawned = spawn_tail(tail, qdir)
                    print("[queue] EMPTY -- firing the TAIL %s -> %s (always-training "
                          "stays true without a human in the loop; refill the queue "
                          "proper when there is a real question)"
                          % (os.path.basename(tail), os.path.basename(spawned)), flush=True)
                    continue
                print("[queue] EMPTY and NO TAIL -- the standing rule says the queue is "
                      "never empty; refill it (re-checking in 10 min)", flush=True)
                if a.once:
                    return
                time.sleep(600)
                continue
            cfg = pending[0]
            # PARSE FIRST.  A yaml typo used to cost a full queue slot to
            # discover: the trainer started, died on the config, and the entry
            # was filed in done/ next to the runs that worked.  Two arms went
            # that way on 2026-08-19 (an unquoted `notes:` containing ": ").
            # Catching it here costs nothing and names the file and the line.
            try:
                import yaml
                with open(cfg) as f:
                    yaml.safe_load(f)
            except Exception as e:
                print("[queue] UNPARSEABLE %s -- moved to failed/, NOT run:\n  %s"
                      % (os.path.basename(cfg), e), flush=True)
                with open(os.path.join(qdir, "LOG.md"), "a") as f:
                    f.write("- %s: %s UNPARSEABLE, not run\n"
                            % (time.strftime("%Y-%m-%d %H:%M"),
                               os.path.basename(cfg)))
                os.rename(cfg, os.path.join(failed, os.path.basename(cfg)))
                continue
            t0 = time.time()
            rc = run_one(cfg, qdir, globs)
            line = "- %s: %s rc=%d in %.0f min\n" % (
                time.strftime("%Y-%m-%d %H:%M"), os.path.basename(cfg), rc,
                (time.time() - t0) / 60)
            with open(os.path.join(qdir, "LOG.md"), "a") as f:
                f.write(line)
            print("[queue] " + line.strip(), flush=True)
            # done/ means IT WORKED.  A nonzero rc goes to failed/, so a
            # dead arm is visibly distinct from a finished one instead of
            # being discovered by reading LOG.md line by line.
            os.rename(cfg, os.path.join(done if rc == 0 else failed,
                                        os.path.basename(cfg)))
            if a.once:
                return
    finally:
        with open(os.path.join(lock, "owner")) as f:
            pass
        os.remove(os.path.join(lock, "owner"))
        os.rmdir(lock)


if __name__ == "__main__":
    main()
