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
contrib/lichess/gcp-create.sh <LICHESS_BOT_TOKEN>
```

That creates the VM, installs pypy3 + sunfish + lichess-bot, and starts a
systemd service that survives reboots and restarts on crashes.

## Option B: Oracle Cloud free tier (Ampere A1)

Create an always-free ARM instance (up to 4 cores / 24GB) with Ubuntu in the
Oracle Cloud console, then on the VM:

```bash
curl -sL https://raw.githubusercontent.com/thomasahle/sunfish/master/contrib/lichess/setup.sh \
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
```

Notes:
- The engine runs under pypy3 automatically (sunfish.py picks the fastest
  available interpreter), with pondering enabled.
- `config.yml` here is a template; `setup.sh` copies it and fills in the
  token. It accepts casual + rated standard games at bullet/blitz/rapid.
- On a 1GB VM keep `TABLE_SIZE` at `100000` (~100MB); setup.sh also adds 1GB
  of swap as a safety margin.
