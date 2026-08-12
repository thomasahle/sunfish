# Packed-NNUE sunfish as a lichess bot

Deployment bundle for running `sunfish_nnue.py` (through `tools/uci.py`:
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

`setup.sh` installs pypy3 + the pinned lichess-bot bridge **plus
`lichess-bot.patch`** (three fixes from the 2026-08-11 production incident:
games past `challenge.concurrency` are aborted promptly instead of silently
starved, a dead event stream restarts the bot instead of leaving it
deaf-but-online, and a failed chat POST can no longer cancel the engine's
move; plus the 2026-08-12 fix: a replayed `gameStart` for a game already
being served is ignored, and an overflow game the bot cannot abort is named
in an ERROR and left alone, never resigned — a game we are not playing is
not ours to end). The bot-integration CI job applies the same patch, so the tested tree
is the deployed tree; `tests/test_lichess_config.py` keeps the pin, the
patch, and the greeting lengths (lichess drops chat over 140 chars) honest.
`watchdog.sh` + `sunfish-watchdog.timer` are the belt-and-braces layer: if
lichess says a move is pending for us and the journal has been silent for
90 s, the service is restarted (startup resumes live games, so this is safe).

`setup.sh` then runs the
`packed/verify.py` battery ON THE INSTANCE as a hard gate (lane integrity,
incremental == from-scratch, engine == reference, exact antisymmetry) before
enabling the systemd units. What is running is recorded in
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

## What can and cannot cost money here

The A1 always-free shape has **dedicated OCPUs and no credit mechanism**:
game load, pondering, CPU saturation -- none of it can bill. The reachable
paid surfaces are network egress past the free 10 TB/month, storage growth
past the free allowance, and (the one real foot-gun) provisioning a shape
outside the always-free envelope. Defenses, in order:

1. `setup.sh` **statically refuses** to install unless instance metadata
   says `VM.Standard.A1.Flex`, ≤2 OCPUs, ≤12 GB, boot ≤60 GB.
2. `cost_gate.py --daemon` (`sunfish-cost-gate.service`) polls the OCI
   Usage API hourly via instance principals. If the tenancy shows **any**
   month-to-date cost above the threshold (default $0.00), it raises
   `/run/sunfish-costgate` and the bot **declines new challenges** (games
   in progress finish; the bot stays online). Once tripped it stays
   tripped for the month -- a reading that "goes away" is more likely
   amendment lag than a refund -- and auto-clears on a clean month
   rollover. An unreachable Usage API does NOT trip the gate (an API
   outage must not kill the bot) unless it stays dark for 48 h straight.
3. Manual clear after investigating:
   `sudo /opt/lichess-bot/venv/bin/python /opt/sunfish/nnue_4k/lichess/cost_gate.py --clear`
   then `sudo systemctl restart sunfish-cost-gate`.

### Instance-principal prerequisites (run once, at deploy time)

The daemon authenticates as the instance (no keys on disk). Tenancy admin
runs, substituting the compartment OCID:

```
oci iam dynamic-group create --name sunfish-bot-instances \
  --description "sunfish lichess bot instances" \
  --matching-rule "instance.compartment.id = '<COMPARTMENT_OCID>'"
oci iam policy create --name sunfish-bot-usage-read \
  --compartment-id <TENANCY_OCID> \
  --statements '["define tenancy usage-report as ocid1.tenancy.oc1..aaaaaaaaned4fkpkisbwjlr56u7cj63lf3wffbilvqknstgtvzub7vhqkggq", "endorse dynamic-group sunfish-bot-instances to read objects in tenancy usage-report", "allow dynamic-group sunfish-bot-instances to read usage-reports in tenancy"]'
```

(The `define/endorse` pair is Oracle's documented boilerplate for usage
access; the last statement is what `RequestSummarizedUsages` checks.)
