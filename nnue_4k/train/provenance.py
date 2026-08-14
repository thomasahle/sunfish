"""Provenance pinning: a training run that cannot be reproduced cannot be
compared with its own successor (distill_train.py's rule, made a module).

Every run directory gets a PROVENANCE.json holding the git sha (and dirty
state), the seed, torch/python versions, the sha256 of every data file the
run read, and the canonical config hash.  Written BEFORE training starts,
so an aborted run still says what it was.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time


def git_state(repo=None):
    repo = repo or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"sha": sha, "dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        # a bare training dir on the box has no .git; the deployer records the
        # sha in PROVENANCE.txt instead.  Absence is reported, never invented.
        return {"sha": None, "dirty": None}


def file_sha256(path, blocksize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocksize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect(cfg, cfg_hash, data_paths):
    import torch
    return {
        "config_hash": cfg_hash,
        "git": git_state(),
        "seed": cfg.opt.seed,
        "torch": torch.__version__,
        "numpy": __import__("numpy").__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname_class": "box" if os.path.exists(os.path.expanduser("~/sunfish-bench")) else "dev",
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_sha256": {os.path.basename(p): file_sha256(p)
                        for p in data_paths if p and os.path.exists(p)},
    }


def write(run_dir, prov):
    with open(os.path.join(run_dir, "PROVENANCE.json"), "w") as f:
        json.dump(prov, f, indent=1, sort_keys=True)
