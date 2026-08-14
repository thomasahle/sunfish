"""nnue_4k/train -- the config-driven PyTorch training pipeline for the
packed big-int net family (tiny ternary/quantized nets with PST-shaped
features: ps768, king buckets, bilinear folds, rff sketches).

Modules
  config       dataclass/yaml experiment configs, canonical config hash
  provenance   git sha / seed / torch version / data sha pinning per run
  features     feature extractors (768 piece-square base, kb4/8/16, mirror)
  data         unified Dataset over dump/.pkl/.npz/FEN sources, FEN-key splits
  model        the net family, antisymmetric BY CONSTRUCTION
  constraints  satpen, ternary STE, phase caps, weight clamps as reusables
  train        the training loop: deterministic, resumable, pinned val
  export       checkpoint -> payload/.sfnn -> entry splice -> pack.sh bytes
  verify_export  bit-exactness: quantized model == packed evaluation
  queue_runner   serial TRAINQUEUE consumer (trainings only), box-aware

House rules carried by this package (see MEASUREMENTS.md for the receipts):
  * val NEVER gates landing -- play does.  Early kill exists only for
    obviously-broken runs (non-finite loss, or worse than the do-nothing
    anchor after warmup).
  * satpen is default-ON (the kbbil collapse: saturation is free in val,
    play pays for it).
  * every run is provenance-pinned and reproducible or it does not count.
"""
