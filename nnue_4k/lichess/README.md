# Packed-NNUE sunfish as a lichess bot

Deployment bundle for running `sunfish_packed.py` (through `tools/uci.py`:
pondering, Hash/TABLE_SIZE, FEN positions) as its own lichess bot on an
Oracle always-free ARM instance (A1, 2 OCPU / 12 GB, aarch64, Ubuntu).

## Deploy (once Thomas's instance + bot token exist)

1. Freeze the build: pick the crowned net and the engine commit, tag it
   (`git tag lichess-packed-vN <commit>`), fill `ENGINE_TAG` and
   `NET_SHA256` (of the `.sfnn` file) into `setup.sh`, commit that too. Deployments run tagged
   builds only -- `setup.sh` refuses placeholders and verifies the net
   hash.
2. `scp` the `.sfnn` net file to the instance.
3. As root: `setup.sh <LICHESS_BOT_TOKEN> <net.pickle>`.

`setup.sh` installs pypy3 + the pinned lichess-bot bridge, then runs the
`packed/verify.py` battery ON THE INSTANCE as a hard gate (lane integrity,
incremental == from-scratch, engine == reference, exact antisymmetry) before
enabling the systemd unit. What is running is recorded in
`/opt/sunfish/nnue_4k/DEPLOYED.txt`.

## aarch64 status

The packed representation is pure Python big-int arithmetic; there is
nothing architecture-specific to port. Verified at prep time on
pypy3 7.3.23 / arm64 (macOS): the full battery is green for both the plain
and the extended (bilinear + tail + phase) evaluation paths. The setup gate
re-proves it on the Linux/aarch64 deploy image. Benchmark nps on the
instance once it exists (`packed/bench.py`, at nice 19) before choosing
matchmaking TCs.

## Design notes

- No credit-gate machinery: always-free A1 shapes are not CPU-throttled.
  (Classic's gate + its service stay in `contrib/lichess/`, which this
  bundle deliberately does not touch.)
- `TABLE_SIZE: 1000000`: sized to 12 GB (arithmetic in `config.yml`);
  removes the eviction-pressure regime; the root-eviction guard in the
  engine stays as belt-and-braces.
- No bullet in `challenge.time_controls` (deep-iteration overrun class;
  see the armed-deadline fix in both go loops).
- The integration test (`tests/test_bot_integration.py`, `BOT_CI=1`) runs
  this exact stack -- packed engine + tools/uci.py + pinned lichess-bot --
  against the in-process mock server, pondering on.
- FEN glue is proven by `tests/test_packed_fen.py`: full-game round-trip
  of every Position field including the accumulator, perspective flag,
  king-bucket index and piece count; en passant live after FEN load.
