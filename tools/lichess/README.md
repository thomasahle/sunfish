# Running sunfish on lichess

This directory contains everything needed to run [sunfish-engine](https://lichess.org/@/sunfish-engine)
as an always-on lichess bot on a free cloud VM, using the
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) bridge.
The bot only makes outbound connections, so any tiny VM works.

## One-time prerequisites

1. **A lichess bot token**: log in to the bot account and create a token with
   the `bot:play` scope at
   <https://lichess.org/account/oauth/token/create?scopes[]=bot:play&description=sunfish-bot>.

## Option A: Google Cloud free tier (e2-micro)

The e2-micro VM in `us-west1`, `us-central1` or `us-east1` with a ≤30GB
standard disk is in GCP's always-free tier.

```bash
gcloud auth login
gcloud config set project <your-project-id>   # project must have billing enabled
tools/lichess/gcp-create.sh <LICHESS_BOT_TOKEN>
```

That creates the VM, installs pypy3 + sunfish + lichess-bot, and starts a
systemd service that survives reboots and restarts on crashes.

## Option B: Oracle Cloud free tier (Ampere A1)

Create an always-free ARM instance (up to 4 cores / 24GB) with Ubuntu in the
Oracle Cloud console, then on the VM:

```bash
curl -sL https://raw.githubusercontent.com/thomasahle/sunfish/master/tools/lichess/setup.sh \
  | sudo bash -s -- <LICHESS_BOT_TOKEN>
```

With 24GB of RAM you can raise `TABLE_SIZE` in `/opt/lichess-bot/config.yml`
to `1000000` (~1GB) and restart: `sudo systemctl restart sunfish-lichess`.

## Operations

```bash
systemctl status sunfish-lichess        # is it running?
journalctl -u sunfish-lichess -f        # live logs / games
sudo systemctl restart sunfish-lichess  # after config changes
cd /opt/sunfish && sudo -u sunfish git pull   # update the engine

systemctl status sunfish-credit-gate    # the CPU-credit gate
python3 /opt/sunfish/tools/lichess/cpu_credit.py --status   # gate snapshot
ls -l /run/sunfish-throttled            # exists => declining new challenges
```

Notes:
- The engine runs under pypy3 automatically (sunfish.py picks the fastest
  available interpreter), with pondering enabled.
- `config.yml` here is a template; `setup.sh` copies it and fills in the
  token. It accepts casual + rated standard games at bullet/blitz/rapid.
- lichess-bot is pinned and patched: `setup.sh` checks out the same commit
  the bot-integration CI tests, then applies `lichess-bot.patch` (2026-08-11
  production fixes: overflow games beyond `challenge.concurrency` are
  aborted promptly instead of silently starved; a dead event stream
  restarts the bot instead of leaving it deaf-but-online; a failed chat
  POST can no longer cancel the engine's move). `watchdog.sh` +
  `sunfish-watchdog.timer` restart the service if lichess says a move is
  pending for us while the journal sits silent for 90 s.
- On a 1GB VM keep `TABLE_SIZE` at `50000` (~120MB peak RSS); setup.sh also
  adds 1GB of swap as a safety margin. See "Sizing TABLE_SIZE" below.

## Running on a shared-core VM (e2-micro)

An `e2-micro` shows two vCPUs but is only entitled to **0.25 vCPU sustained**.
With pondering the engine computes on the opponent's turn too, so a game draws
far more than that. Three separate mechanisms keep the bot honest on such a
box; they address genuinely different failure modes, so keep all three.

### 1. Priority separation (the important one)

lichess-bot's event loop must stay responsive enough to read the opponent's
move off the socket *while the engine is pondering*. At equal priority a deep
ponder search can starve it — in production this showed up as a 23-second
stall that flagged an otherwise-won game with 26 seconds on the clock.

- `sunfish-lichess.service` sets `CPUWeight=200` for the bot's cgroup.
- `config.yml` launches the engine through `nice -n 10` (lichess-bot's
  `interpreter` option), so the engine sits well below the event loop.
  `nice` execs `sunfish.py`, whose polyglot header then execs pypy3, so the
  niceness is inherited and the pypy3/python3 fallback still works.

### 2. Sizing TABLE_SIZE from measured memory

An engine that swaps mid-search is slow in a way no scheduling fix repairs.
Measured under pypy3, each transposition entry costs **~0.8 kB** resident, and
`TABLE_SIZE` bounds `tp_score` and `tp_move` *independently* — so the tables
can hold `2 * TABLE_SIZE` entries:

| TABLE_SIZE | max entries | tables  | + pypy3 baseline (~36 MB) |
|-----------:|------------:|--------:|--------------------------:|
| 50 000     | 100 000     | ~80 MB  | **~120 MB peak RSS**      |
| 100 000    | 200 000     | ~160 MB | ~196 MB peak RSS          |
| 300 000    | 600 000     | ~480 MB | ~516 MB peak RSS          |
| 1 000 000  | 2 000 000   | ~1.6 GB | needs a 4 GB+ box         |

The 1 GB VM has ~970 MB usable and already runs lichess-bot's multiprocessing
pool (~150 MB across six processes). The production box was found running
**`TABLE_SIZE: 300000`** — roughly **516 MB peak engine RSS**, which simply
does not fit alongside the bot in 969 MB. It sat with **~245 MB paged out
while completely idle** and accumulated **467 s of full IO-pressure stall**
(`/proc/pressure/io`) over 3.75 days. Hence `50000` here.

> **Keep this template and the live `config.yml` in sync.** `setup.sh` only
> copies the template on a *fresh* install, so later hand-edits on the VM
> silently diverge. The 300000 above was discovered only when deploying; the
> template still claimed 100000, and the live box had also flipped
> `accept_bot` to `true`. Before sizing anything from this file, check the
> real value:
>
> ```bash
> sudo grep -nE "TABLE_SIZE|accept_bot|ponder:" /opt/lichess-bot/config.yml
> ```

### 3. The CPU-credit gate

`cpu_credit.py --daemon` (unit: `sunfish-credit-gate.service`) samples
`/proc/stat`, models a burst-credit deficit, and maintains the flag file
`/run/sunfish-throttled`. While that file exists, lichess-bot **declines new
challenges** — it never interrupts a game in progress.

The bot side is `extra_game_handlers.py`, which implements lichess-bot's
supported `is_supported_extra()` hook. Nothing in `/opt/lichess-bot` is
patched, so `git pull` there keeps working.

**Read `cpu_credit.py`'s module docstring before tuning it.** GCE exposes no
credit metric (unlike AWS `CPUCreditBalance`), so the bucket size is
*modelled*, not measured, and the defaults sit two orders of magnitude above
anything this instance has ever reached — the gate is a safety net, not a
duty-cycle limiter. The one *measured* signal is sustained CPU steal, which
closes the gate on its own regardless of the model.

Both the flag and the hook fail **open**: a missing, unreadable or stale
(>5 min) flag means "accept". A dead estimator must never strand the bot
offline.

Declines use lichess's `generic` reason ("Challenge declined"), because
`lib/model.py` hardcodes that for this hook. If you would rather send `later`
("This is not a good time for me, please ask again later"), which is closer to
what the gate means, patch the one line locally — and re-apply it after any
lichess-bot upgrade:

```bash
sudo -u sunfish sed -i \
  's/self.decline_due_to(is_supported_extra(self), "generic")/self.decline_due_to(is_supported_extra(self), "later")/' \
  /opt/lichess-bot/lib/model.py
```

## Deploying these changes to a running bot

The scripted way, from your own machine (idle-gated restart via the lichess
API, drift checks, engine smoke test; also handles the NNUE bot and PyPI
release tags):

```bash
tools/deploy.sh classic user@host   # this bundle
tools/deploy.sh nnue                # nnue_4k/lichess on the Oracle box
tools/deploy.sh nnue --check        # report-only
tools/deploy.sh pypi v2027         # push a release tag; CI publishes
```

The manual steps it automates:

```bash
# 1. Pick up the new engine + contrib files.
#    If this fails with "insufficient permission for adding an object", a
#    previous update was run as root and left root-owned objects behind:
#      sudo chown -R sunfish:sunfish /opt/sunfish
cd /opt/sunfish && sudo -u sunfish git pull

# 2. Install the gate daemon and the updated bot unit
sudo cp tools/lichess/sunfish-credit-gate.service /etc/systemd/system/
sudo cp tools/lichess/sunfish-lichess.service     /etc/systemd/system/
sudo cp tools/lichess/extra_game_handlers.py      /opt/lichess-bot/
sudo chown sunfish /opt/lichess-bot/extra_game_handlers.py
sudo systemctl daemon-reload

# 3. Start the gate first and confirm it is sane BEFORE touching the bot
sudo systemctl enable --now sunfish-credit-gate
sleep 30
python3 /opt/sunfish/tools/lichess/cpu_credit.py --status
#   expect: "gate_open": true, small deficit, "throttled_now": false
ls -l /run/sunfish-throttled     # should NOT exist on a healthy box

# 4. Apply the engine settings (TABLE_SIZE + nice) to the bot's config.
#    Edit /opt/lichess-bot/config.yml by hand -- do NOT re-copy the template,
#    it would overwrite your token:
#      uci_options.TABLE_SIZE: 50000
#      engine.interpreter: "nice"
#      engine.interpreter_options: ["-n", "10"]

# 5. Restart the bot when no game is in progress
systemctl status sunfish-lichess     # check the log for an active game first
sudo systemctl restart sunfish-lichess

# 6. Verify
journalctl -u sunfish-lichess -f     # play one game against the bot
ps -o pid,ni,rss,comm -C pypy3       # engine should show NI=10
free -m                              # swap in use should fall over time
```

### Rollback

```bash
# Gate off, priorities and table size back to the old values
sudo systemctl disable --now sunfish-credit-gate
sudo rm -f /run/sunfish-throttled
sudo rm -f /etc/systemd/system/sunfish-credit-gate.service

# Restore the no-op hook that ships with lichess-bot
sudo -u sunfish tee /opt/lichess-bot/extra_game_handlers.py >/dev/null <<'EOF'
def game_specific_options(game):
    return {}


def is_supported_extra(challenge):
    return True
EOF

# Revert config.yml: restore the previous TABLE_SIZE (the production box was
# on 300000, not the template's 100000 -- check your backup), drop
# interpreter/interpreter_options.
# Revert the unit (drop CPUWeight):
cd /opt/sunfish && sudo -u sunfish git checkout HEAD~1 -- tools/lichess/
sudo cp tools/lichess/sunfish-lichess.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart sunfish-lichess
```

Removing `/run/sunfish-throttled` is enough to un-gate the bot immediately;
stopping the daemon alone also works, since the hook expires a flag older than
five minutes.
