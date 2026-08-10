#!/bin/bash
# Sets up the packed-NNUE sunfish lichess bot on a fresh Ubuntu ARM instance
# (Oracle always-free A1: 2 OCPU / 12 GB, aarch64).  No credit-gate machinery:
# always-free shapes do not throttle.
#
# Usage (as root, on the instance):
#   setup.sh <LICHESS_BOT_TOKEN> <NET_FILE>
#
#   <NET_FILE> is the packed net pickle (scp it to the instance first).
#   Its sha256 must match NET_SHA256 below -- the deployed engine is a
#   frozen, tagged build and the net is part of that freeze.
set -euo pipefail

# ---- FROZEN BUILD PIN (fill at freeze time; deployment refuses to run
# ---- with placeholders left in)
ENGINE_TAG="FILL_ME_TAG"          # git tag in thomasahle/sunfish, packed-nnue lane
NET_SHA256="FILL_ME_SHA256"       # sha256 of the packed net pickle

TOKEN="${1:-${LICHESS_TOKEN:-}}"
NET="${2:-}"
if [ -z "$TOKEN" ] || [ -z "$NET" ]; then
    echo "Usage: setup.sh <LICHESS_BOT_TOKEN> <NET_FILE>" >&2
    echo "Token: https://lichess.org/account/oauth/token/create?scopes[]=bot:play" >&2
    exit 1
fi
case "$ENGINE_TAG$NET_SHA256" in *FILL_ME*)
    echo "ERROR: freeze pins not filled in (ENGINE_TAG/NET_SHA256)." >&2
    echo "Deployments run tagged builds only." >&2
    exit 1
esac
echo "$NET_SHA256  $NET" | sha256sum -c - || {
    echo "ERROR: net file does not match the frozen NET_SHA256." >&2; exit 1; }

apt-get update -qq
apt-get install -y -qq git pypy3 python3-venv python3-pip

id -u sunfish &>/dev/null || useradd -r -m sunfish

# The engine, at the frozen tag exactly.
[ -d /opt/sunfish-packed ] || git clone https://github.com/thomasahle/sunfish /opt/sunfish-packed
git -C /opt/sunfish-packed fetch -q --tags origin
git -C /opt/sunfish-packed checkout -q -f "$ENGINE_TAG"

install -m 644 "$NET" /opt/sunfish-packed/net.pickle

# ---- aarch64 correctness gate: the packed big-int arithmetic is pure
# Python and the verify battery proves lane integrity, incremental ==
# from-scratch, engine == reference and exact antisymmetry ON THIS
# MACHINE before the bot is allowed to exist.  (Already green on
# pypy3/arm64 macOS at prep time; this re-proves it on the deploy image.)
SF_NET=/opt/sunfish-packed/net.pickle \
    pypy3 /opt/sunfish-packed/packed/verify.py \
    /opt/sunfish-packed/sunfish_packed.py /opt/sunfish-packed/net.pickle 120 40

# The bridge, pinned to the commit the integration test runs against.
LICHESS_BOT_COMMIT=bedd1d9e86a8c4c96319490533e4e20fe63d1ac8
[ -d /opt/lichess-bot ] || git clone https://github.com/lichess-bot-devs/lichess-bot /opt/lichess-bot
git -C /opt/lichess-bot fetch -q origin "$LICHESS_BOT_COMMIT"
git -C /opt/lichess-bot checkout -q -f "$LICHESS_BOT_COMMIT"

python3 -m venv /opt/lichess-bot/venv
/opt/lichess-bot/venv/bin/pip install -q -r /opt/lichess-bot/requirements.txt

cp /opt/sunfish-packed/contrib/lichess-packed/config.yml /opt/lichess-bot/config.yml
sed -i "s/YOUR_TOKEN_HERE/$TOKEN/" /opt/lichess-bot/config.yml
chmod 600 /opt/lichess-bot/config.yml

# Record exactly what is deployed (never hide what runs).
{ echo "engine tag:  $ENGINE_TAG ($(git -C /opt/sunfish-packed rev-parse HEAD))"
  echo "net sha256:  $NET_SHA256"
  echo "bridge:      $LICHESS_BOT_COMMIT"
  echo "deployed:    $(date -u)"
} > /opt/sunfish-packed/DEPLOYED.txt

chown -R sunfish:sunfish /opt/lichess-bot /opt/sunfish-packed

cp /opt/sunfish-packed/contrib/lichess-packed/sunfish-packed.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now sunfish-packed

echo
echo "Done. Status:   systemctl status sunfish-packed"
echo "Logs:           journalctl -u sunfish-packed -f"
echo "Deployed build: cat /opt/sunfish-packed/DEPLOYED.txt"
