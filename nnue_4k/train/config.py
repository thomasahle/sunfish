"""Experiment configs: dataclasses, yaml load/save, canonical hash.

A run is DEFINED by its config: the same config (plus the same data shas and
the same code sha) must reproduce the same run.  The hash of the canonical
JSON form is stamped into PROVENANCE.json and into the run directory name,
so two runs can be compared by eye.
"""
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class DataCfg:
    source: str = ""            # .jsonl.zst dump | legacy .pkl cache | labeled .npz | .binpack
    kind: str = "auto"          # auto | dump | legacy-pkl | npz | binpack
    limit: int = 4_100_000
    cpmax: int = 1000           # drop |cp| above this (train_packed convention)
    quiet: int = 0              # 1 = quiet-position filter at parse time
    workers: int = 0            # parse processes (box: <= 8, labeller-class)
    cache: str = ""             # tensor cache (.npz new format, .pkl legacy read-only)
    split: str = "fenkey"       # fenkey | legacy-perm  (see data.py)
    split_seed: int = 20260813  # FEN-key hash seed (house rule, distill_train.py)
    val_mod: int = 20           # fenkey: hash %% val_mod == 0 is val (20 -> 5%)
    valn: int = 0               # legacy-perm: --valn pinning (0 = permute all)


@dataclass
class ModelCfg:
    arch: str = "residual"      # residual | ml2 | cb | lowrank (see model.py)
    N: int = 4                  # hidden units per perspective
    kb: int = 1                 # own-king buckets (1/4/8/16)
    factor: int = 1             # virtual per-piece-type features (folded at export)
    base: str = "mat"           # fixed base under the residual: mat | pst
    ternary: float = 0.0        # replnet STE threshold tau (0 = off)
    segs: int = 1               # crelu segments (1 = clipped ReLU)
    clampcp: int = 600          # residual clip, the operator the engine runs
    nb: int = 0                 # bilinear lanes per perspective (0 = off)
    bm: int = 4                 # bilinear groups m (m=4 folds mod 2^64-1)
    nb2: int = 0                # two-block bilinear
    baff: int = 0               # affine bilinear offsets
    tailw: int = 0              # odd-symmetrized narrow tail width
    phase: int = 0              # material-phase output buckets (0 = off)
    rff: int = 0                # phase-sketch lanes (0 = off)
    rffsigma: float = 0.5
    # --- trained structure (structures.py); certified before training
    cb_k: int = 32              # arch=cb: codebook entries K
    cb_block: int = 8           # arch=cb: features per block (must divide 768)
    cb_temp: float = 0.5        # arch=cb: soft-backward temperature (per element)
    cb_cmax: int = 1            # arch=cb: codebook grid (1 = ternary; <= 5 certifies)
    lr_rank: int = 1            # arch=lowrank: rank of U@V
    lr_wmax: int = 1            # arch=lowrank: composite clip (1 = ternary)
    gridste: int = 0            # ternary path: snap the OUTPUT weights and the
    #                             biases to the integer digits the payload
    #                             stores (v = 32*g/2^s, b = bd/(32*g)) inside
    #                             forward, by STE.  With ternary weights (grid
    #                             already) and u2grid, this is FULL exported
    #                             fidelity: nothing the artifact rounds is
    #                             trained at a precision it does not have.
    u2grid: int = 0             # arch=ml2: snap u2 to the CERTIFIED integer
    #                             read-out grid inside forward (STE), at the
    #                             export scale.  0 = free float, which is what
    #                             let the 0.01280 net train a layer 2 the
    #                             engine then rounded to zero (MEASUREMENTS
    #                             2026-08-15).  1 = train against the real
    #                             resolution.


@dataclass
class LossCfg:
    sigK: float = 400.0         # cp -> win-prob scale
    losspow: float = 2.0        # |sig(pred)-sig(y)|^p
    satpen: float = 0.03        # saturation penalty -- DEFAULT ON (kbbil lesson)
    satthresh: float = 480.0    # cp where the penalty starts
    l1: float = 0.0             # sparsity pressure on pre-ternarization |u|
    rate: float = 0.0           # payload-rate pressure, val-units per est. BYTE
    rate_T: float = 8.0         # softness of the rate term's trit occupancy


@dataclass
class OptCfg:
    epochs: int = 40
    batch: int = 8192
    lr: float = 3e-3
    weight_decay: float = 1e-5
    wclip: float = 1.0          # effective-weight clamp after every step
    phasecap: float = 0.0       # project phase scales into [1/cap, cap]
    seed: int = 0               # torch + shuffle seed (train_packed pinned 0)
    threads: int = 0            # torch CPU threads (0 = torch default)


@dataclass
class TrainConfig:
    name: str = "run"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    opt: OptCfg = field(default_factory=OptCfg)
    out_dir: str = ""           # default: runs/<name> under nnue_4k/train
    notes: str = ""


def to_dict(cfg):
    return dataclasses.asdict(cfg)


def from_dict(d):
    d = dict(d)
    kw = {}
    for name, cls in (("data", DataCfg), ("model", ModelCfg),
                      ("loss", LossCfg), ("opt", OptCfg)):
        sub = dict(d.pop(name, {}))
        known = {f.name for f in dataclasses.fields(cls)}
        bad = set(sub) - known
        if bad:
            raise ValueError("unknown %s keys: %s" % (name, sorted(bad)))
        kw[name] = cls(**sub)
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    bad = set(d) - known
    if bad:
        raise ValueError("unknown config keys: %s" % sorted(bad))
    return TrainConfig(**d, **kw)


def load(path):
    with open(path) as f:
        text = f.read()
    if str(path).endswith((".yml", ".yaml")):
        import yaml
        return from_dict(yaml.safe_load(text))
    return from_dict(json.loads(text))


def save(path, cfg):
    with open(path, "w") as f:
        if str(path).endswith((".yml", ".yaml")):
            import yaml
            yaml.safe_dump(to_dict(cfg), f, sort_keys=True)
        else:
            json.dump(to_dict(cfg), f, indent=1, sort_keys=True)


def config_hash(cfg):
    """Canonical hash of the experiment definition.  out_dir and notes are
    presentation, not definition; data.workers and opt.threads change wall
    time, not the function being optimised -- all excluded."""
    d = to_dict(cfg)
    d.pop("out_dir", None)
    d.pop("notes", None)
    d["data"].pop("workers", None)
    d["opt"].pop("threads", None)
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
