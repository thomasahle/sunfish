#!/usr/bin/env python3
"""The training loop: deterministic, resumable, provenance-pinned.

VAL DOES NOT GATE LANDING -- PLAY DOES.  (Pre-registered, MEASUREMENTS.md
2026-08-14: "val gates only the recipe against its own float baseline --
val LANDS NOTHING."  Held-out loss mis-ranked C2 by 5.9% while it lost 94
Elo, and the best-val net ever trained here collapsed -118 in play.)  The
early-kill below exists ONLY for obviously-broken runs: a non-finite loss,
or val still worse than the do-nothing anchor after warmup.  It must never
be tightened into a quality gate.

usage:
  train.py CONFIG.yaml [--resume] [--out-dir DIR]
  train.py --repro-arm1 DATA_OR_CACHE [--out-dir DIR]   # the v1 arm 1 recipe

--repro-arm1 is the pipeline's own validation instrument: the REPLNET v1
arm 1 recipe (l1=0.001, tau=0.85, 40 epochs, legacy split, seed 0) with the
RNG streams aligned to train_packed.py, so the val series is directly
comparable to the box ledger (winner: val 0.01385 @ 59.6% zeros).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfgmod           # noqa: E402
import constraints                # noqa: E402
import data as datamod            # noqa: E402
import export as exportmod        # noqa: E402
import features                   # noqa: E402
import provenance                 # noqa: E402
import field_budget               # noqa: E402
from model import build_model, lambda_loss  # noqa: E402


def repro_arm1_config(source):
    """REPLNET v1 arm 1, exactly as run on the box (RUN.log + process args):
    train_packed.py --N 4 --base mat --ternary 0.85 --l1 0.001 --satpen 0.03
    --satthresh 480 --clampcp 600 --quiet 0 --cpmax 1000 --epochs 40
    --batch 8192 --lr 3e-3 --limit 4100000, seed 0, legacy split."""
    return cfgmod.TrainConfig(
        name="repro_arm1",
        data=cfgmod.DataCfg(source=source, limit=4_100_000, cpmax=1000, quiet=0,
                            split="legacy-perm", valn=0),
        model=cfgmod.ModelCfg(N=4, kb=1, factor=1, base="mat", ternary=0.85,
                              clampcp=600),
        loss=cfgmod.LossCfg(sigK=400.0, losspow=2.0, satpen=0.03,
                            satthresh=480.0, l1=0.001),
        opt=cfgmod.OptCfg(epochs=40, batch=8192, lr=3e-3, seed=0),
        notes="pipeline validation: reproduce v1 arm 1 (ledger val 0.01385 @59.6% zeros); "
              "pre-stated tolerance |dval| <= 0.0002, zeros within +-5 points")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", help="yaml/json TrainConfig")
    p.add_argument("--repro-arm1", metavar="DATA", help="run the pinned arm 1 repro recipe")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dir", default="")
    p.add_argument("--threads", type=int, default=0, help="override opt.threads")
    p.add_argument("--workers", type=int, default=0, help="override data.workers")
    a = p.parse_args()

    if a.repro_arm1:
        cfg = repro_arm1_config(a.repro_arm1)
    elif a.config:
        cfg = cfgmod.load(a.config)
    else:
        p.error("need CONFIG.yaml or --repro-arm1 DATA")
    if a.out_dir:
        cfg.out_dir = a.out_dir
    if a.threads:
        cfg.opt.threads = a.threads
    if a.workers:
        cfg.data.workers = a.workers
    if cfg.opt.threads:
        torch.set_num_threads(cfg.opt.threads)

    chash = cfgmod.config_hash(cfg)
    run_dir = cfg.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "runs", "%s-%s" % (cfg.name, chash))
    os.makedirs(run_dir, exist_ok=True)
    cfgmod.save(os.path.join(run_dir, "config.yaml"), cfg)
    log_path = os.path.join(run_dir, "metrics.jsonl")

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- data (before torch seeding: parsing consumes no torch RNG,
    # matching train_packed.py's order)
    t0 = time.time()
    ds = datamod.load(cfg.data)
    print("%d positions in %.0fs" % (len(ds), time.time() - t0), flush=True)
    base = ds.base(cfg.model.base)

    # ---- determinism.  ONE python Random drives split + epoch shuffles, in
    # the exact call order train_packed.py drew from the global stream.
    torch.manual_seed(cfg.opt.seed)
    rng = random.Random(cfg.opt.seed)
    # data.perm_seed pins the SPLIT so a seed sweep varies initialisation and
    # batch order and NOT the validation set.  Without it opt.seed feeds the
    # split permutation too, and two seeds draw val sets that overlap 5% --
    # i.e. the experiment changes its own measuring stick (run 76, 2026-08-15).
    # Unset (-1) keeps the historical single-stream order bit for bit.
    split_rng = rng if cfg.data.perm_seed < 0 else random.Random(cfg.data.perm_seed)
    train_ids, val_ids = datamod.make_split(ds, cfg.data, split_rng)
    vsha = datamod.val_sha(ds, val_ids)
    print("split %s: %d train / %d val  val-sha %s"
          % (cfg.data.split, len(train_ids), len(val_ids), vsha), flush=True)

    prov = provenance.collect(cfg, chash, [cfg.data.source, cfg.data.cache])
    prov["val_sha"] = vsha
    prov["n_train"], prov["n_val"] = len(train_ids), len(val_ids)
    provenance.write(run_dir, prov)

    cert = field_budget.certify_or_raise(cfg.model)   # uncertifiable configs never train
    if cert is not None:
        with open(os.path.join(run_dir, "certificate.json"), "w") as f:
            f.write(cert.to_json())
    model = build_model(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.opt.lr,
                            weight_decay=cfg.opt.weight_decay)
    if cfg.opt.sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.opt.epochs)
    elif cfg.opt.sched == "linear":
        sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=0.0, total_iters=cfg.opt.epochs)
    else:
        raise SystemExit("unknown opt.sched %r (want 'cosine' or 'linear')" % cfg.opt.sched)
    print("optimizer AdamW(lr=%g, wd=%g)  schedule %s over %d epochs"
          % (cfg.opt.lr, cfg.opt.weight_decay, cfg.opt.sched, cfg.opt.epochs), flush=True)

    start_epoch, best = 0, float("inf")
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        if ck["config_hash"] != chash:
            raise SystemExit("checkpoint %s is from config %s, not %s -- refusing "
                             "to resume a different experiment"
                             % (ckpt_path, ck["config_hash"], chash))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        rng.setstate(ck["py_rng"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print("resumed at epoch %d (best val %.5f)" % (start_epoch, best), flush=True)

    ext = features.extractor_for(cfg.model.kb)
    K, CLAMP = cfg.loss.sigK, float(cfg.model.clampcp)
    out_net = os.path.join(run_dir, "best.pickle")

    # anchor rows: what the val split costs with no net at all.  These are
    # also the early-kill reference (see module docstring).
    vc = torch.tensor(val_ids)
    vy, vb = ds.y[vc], base[vc]
    # THE TARGET IS THE TRAINED TARGET, in val and in the anchors alike.  A
    # lambda<1 arm is trained toward outcomes, so scoring it against a pure-cp
    # target measures the wrong thing -- and the EARLY-KILL anchors, computed
    # in cp space, killed a correctly-training lam=0 arm at epoch 2 in the
    # smoke.  Blending both keeps the tripwire meaningful.  NOTE: val is then
    # NOT comparable ACROSS lambda arms (different targets); the judge is the
    # selector, and the lambda is printed on every val line so nobody stacks
    # them up later by mistake.
    _lam = cfg.loss.lam
    sy = torch.sigmoid(vy / K)
    if _lam < 1.0:
        sy = _lam * sy + (1.0 - _lam) * ds.outcome[vc]
    zero_anchor = ((torch.sigmoid(vy * 0) - sy) ** 2).mean().item()
    base_anchor = ((torch.sigmoid(vb / K) - sy) ** 2).mean().item()
    print("val anchors: zero %.5f  %s %.5f  (val %d positions, lam %g)"
          % (zero_anchor, cfg.model.base, base_anchor, len(val_ids), _lam), flush=True)

    for epoch in range(start_epoch, cfg.opt.epochs):
        model.train()
        tl = tn = 0
        for fi, mi, fo, c in datamod.batches(ds, train_ids, cfg.opt.batch, ext, rng):
            pred = model(fi, mi, fo, base[c])
            oc = ds.outcome[c] if getattr(ds, "outcome", None) is not None else None
            if cfg.loss.lam < 1.0 and oc is None:
                raise SystemExit("loss.lam=%g needs a game-outcome channel and this "
                                 "corpus has none -- refusing to silently train the "
                                 "lam=1 control under a lambda label"
                                 % cfg.loss.lam)
            loss = lambda_loss(pred, ds.y[c], oc, K, cfg.loss.losspow, cfg.loss.lam)
            if cfg.loss.satpen:
                loss = loss + constraints.saturation_penalty(
                    model.pre, cfg.loss.satpen, cfg.loss.satthresh)
            if cfg.loss.l1:
                loss = loss + constraints.l1_pressure(model._u, cfg.loss.l1)
            if cfg.loss.rate:
                # ternary-only by construction: _u exists iff the STE ran
                loss = loss + constraints.rate_penalty(
                    model._u, cfg.model.ternary, cfg.loss.rate, cfg.loss.rate_T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.clamp_weights(cfg.opt.wclip)
            if cfg.opt.phasecap and cfg.model.phase:
                constraints.phasecap_(model.s, cfg.opt.phasecap)
            tl += loss.item() * len(c)
            tn += len(c)
        sched.step()

        model.eval()
        vl = vn = mae = sat = 0
        with torch.no_grad():
            for fi, mi, fo, c in datamod.batches(ds, val_ids, cfg.opt.batch, ext):
                pred = model(fi, mi, fo, base[c])
                y = ds.y[c]
                t = torch.sigmoid(y / K)
                if _lam < 1.0:
                    t = _lam * t + (1.0 - _lam) * ds.outcome[c]
                se = (torch.sigmoid(pred / K) - t) ** 2
                vl += se.mean().item() * len(c)
                mae += (pred - y).abs().clamp(max=1000).mean().item() * len(c)
                sat += ((pred - base[c]).abs() >= CLAMP - 0.5).sum().item()
                vn += len(c)
        val = vl / vn
        tag = ""
        if val < best:
            best = val
            info = exportmod.export_model(model, cfg, out_net)
            tag = "  -> wrote %s (%s)" % (out_net, info)
        print("epoch %d: train %.5f  val %.5f  val-MAE %.0f cp  clip-saturated %.2f%%%s"
              % (epoch, tl / tn, val, mae / vn, 100.0 * sat / vn, tag), flush=True)
        log({"epoch": epoch, "train": tl / tn, "val": val, "mae": mae / vn,
             "sat": sat / vn, "best": best, "time": time.time() - t0})

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "torch_rng": torch.get_rng_state(),
                    "py_rng": rng.getstate(), "epoch": epoch, "best": best,
                    "config_hash": chash}, ckpt_path)

        # EARLY-KILL: obviously-broken runs only (never a quality gate)
        broken = not math.isfinite(val) or (epoch >= 2 and val > min(zero_anchor, base_anchor))
        if broken:
            print("EARLY-KILL: val %.5f vs anchors zero %.5f / %s %.5f at epoch %d "
                  "-- the run is broken, not merely weak.  Stopping."
                  % (val, zero_anchor, cfg.model.base, base_anchor, epoch), flush=True)
            log({"early_kill": True, "epoch": epoch, "val": val})
            break

    print("best val %.5f -> %s" % (best, out_net), flush=True)
    return best


if __name__ == "__main__":
    main()
