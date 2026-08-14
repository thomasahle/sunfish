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

usage: queue_runner.py [--queue-dir DIR] [--once] [--pgn-globs G1:G2:...]
"""
import argparse
import glob
import os
import signal
import subprocess
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GLOBS = os.pathsep.join((
    os.path.expanduser("~/sunfish-bench/*.pgn"),
    os.path.expanduser("~/sunfish-bench/*/*.pgn"),
))


def forfeit_count(globs):
    n = 0
    for g in globs:
        for path in glob.glob(g):
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


def run_one(cfg_path, qdir, globs):
    name = os.path.basename(cfg_path)
    run_dir = os.path.join(_here, "runs", name.rsplit(".", 1)[0])
    os.makedirs(run_dir, exist_ok=True)
    base = forfeit_count(globs)
    print("[queue] starting %s (forfeit baseline %d)" % (name, base), flush=True)
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
                        % (base, now, name, proc.pid))
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
    os.makedirs(done, exist_ok=True)
    globs = [g for g in a.pgn_globs.split(os.pathsep) if g]
    lock = acquire_lock(qdir)
    try:
        while True:
            pending = sorted(p_ for p_ in glob.glob(os.path.join(qdir, "*.yaml")))
            if not pending:
                print("[queue] EMPTY -- the standing rule says the queue is "
                      "never empty; refill it (re-checking in 10 min)", flush=True)
                if a.once:
                    return
                time.sleep(600)
                continue
            cfg = pending[0]
            t0 = time.time()
            rc = run_one(cfg, qdir, globs)
            line = "- %s: %s rc=%d in %.0f min\n" % (
                time.strftime("%Y-%m-%d %H:%M"), os.path.basename(cfg), rc,
                (time.time() - t0) / 60)
            with open(os.path.join(qdir, "LOG.md"), "a") as f:
                f.write(line)
            print("[queue] " + line.strip(), flush=True)
            os.rename(cfg, os.path.join(done, os.path.basename(cfg)))
            if a.once:
                return
    finally:
        with open(os.path.join(lock, "owner")) as f:
            pass
        os.remove(os.path.join(lock, "owner"))
        os.rmdir(lock)


if __name__ == "__main__":
    main()
