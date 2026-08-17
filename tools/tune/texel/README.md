# Texel evaluation tuning

These scripts fit classic Sunfish piece-square tables to positions labelled by
Stockfish:

- `texel_data.py` samples positions from PGNs and creates a NumPy data set.
- `texel_tune.py` fits one 384-entry piece-square evaluation.
- `texel_taper.py` experiments with separate middlegame and endgame tables.

By default `texel_data.py` reads PGNs from `tools/tune/arena/` and invokes
`stockfish` from `PATH`. Set `ARENA` or `STOCKFISH` to override either path.
The fitting scripts read the current repository's `sunfish.py` as their warm
start, so the checkout path is no longer machine-specific.

```sh
python3 tools/tune/texel/texel_data.py positions.npz
python3 tools/tune/texel/texel_tune.py positions.npz tables.json
```
