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
NET_SHA256="FILL_ME_SHA256"       # sha256 of the .sfnn net file

TOKEN="${1:-${LICHESS_TOKEN:-}}"
NET="${2:-}"
if [ -z "$TOKEN" ] || [ -z "$NET" ]; then
    echo "Usage: setup.sh <LICHESS_BOT_TOKEN> <NET_FILE.sfnn>" >&2
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


# ---- always-free shape assertion: this deployment is designed to be
# incapable of accruing compute cost.  A misprovisioned shape is the one
# mistake that could bill quietly, so refuse to install onto one.
MD="curl -sf -H \"Authorization: Bearer Oracle\" http://169.254.169.254/opc/v2/instance/"
SHAPE=$(eval $MD | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('shape',''))")
OCPUS=$(eval $MD | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('shapeConfig',{}).get('ocpus',0))")
MEMGB=$(eval $MD | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('shapeConfig',{}).get('memoryInGBs',0))")
BOOTGB=$(($(lsblk -b -dn -o SIZE $(findmnt -n -o SOURCE / | sed 's/p\?[0-9]*$//') 2>/dev/null || echo 0) / 1073741824))
if [ "$SHAPE" != "VM.Standard.A1.Flex" ] || \
   ! python3 -c "import sys; sys.exit(0 if float('$OCPUS' or 0) <= 2 and float('$MEMGB' or 0) <= 12 else 1)" || \
   [ "$BOOTGB" -gt 60 ]; then
    echo "ABORT: this instance is NOT inside the always-free envelope:" >&2
    echo "  shape=$SHAPE (need VM.Standard.A1.Flex), ocpus=$OCPUS (<=2)," >&2
    echo "  memory=${MEMGB}GB (<=12), boot=${BOOTGB}GB (<=60)" >&2
    echo "Provision an always-free A1 instance and re-run." >&2
    exit 1
fi
echo "shape check: $SHAPE ocpus=$OCPUS mem=${MEMGB}GB boot=${BOOTGB}GB -- always-free envelope OK"

apt-get update -qq
apt-get install -y -qq git pypy3 python3-venv python3-pip

id -u sunfish &>/dev/null || useradd -r -m sunfish

# The engine, at the frozen tag exactly.
[ -d /opt/sunfish ] || git clone https://github.com/thomasahle/sunfish /opt/sunfish
git -C /opt/sunfish fetch -q --tags origin
git -C /opt/sunfish checkout -q -f "$ENGINE_TAG"

install -m 644 "$NET" /opt/sunfish/nnue_4k/net.sfnn

# ---- aarch64 correctness gate: the packed big-int arithmetic is pure
# Python and the verify battery proves lane integrity, incremental ==
# from-scratch, engine == reference and exact antisymmetry ON THIS
# MACHINE before the bot is allowed to exist.  (Already green on
# pypy3/arm64 macOS at prep time; this re-proves it on the deploy image.)
SF_NET=/opt/sunfish/nnue_4k/net.sfnn \
    pypy3 /opt/sunfish/nnue_4k/packed/verify.py \
    /opt/sunfish/nnue_4k/sunfish_packed.py /opt/sunfish/nnue_4k/net.sfnn 120 40

# The bridge, pinned to the commit the integration test runs against, plus
# the production patch (lichess-bot.patch) the same test applies: overflow
# games are aborted instead of silently starved, a dead event stream
# restarts the bot instead of leaving it deaf, and a failed chat message can
# no longer cancel the engine's move.  Pin + patch = the tested tree.
LICHESS_BOT_COMMIT=bedd1d9e86a8c4c96319490533e4e20fe63d1ac8
[ -d /opt/lichess-bot ] || git clone https://github.com/lichess-bot-devs/lichess-bot /opt/lichess-bot
git -C /opt/lichess-bot fetch -q origin "$LICHESS_BOT_COMMIT"
git -C /opt/lichess-bot checkout -q -f "$LICHESS_BOT_COMMIT"
git -C /opt/lichess-bot apply /opt/sunfish/nnue_4k/lichess/lichess-bot.patch

python3 -m venv /opt/lichess-bot/venv
/opt/lichess-bot/venv/bin/pip install -q -r /opt/lichess-bot/requirements.txt
# the cost tripwire's only dependency (instance-principal auth, no keys)
/opt/lichess-bot/venv/bin/pip install -q oci

cp /opt/sunfish/nnue_4k/lichess/config.yml /opt/lichess-bot/config.yml
# The cost tripwire hook: lichess-bot calls is_supported_extra() from this
# file on every incoming challenge; it declines while /run/sunfish-costgate
# exists.  Replaces the no-op stub that ships with lichess-bot.
cp /opt/sunfish/nnue_4k/lichess/extra_game_handlers.py /opt/lichess-bot/
sed -i "s/YOUR_TOKEN_HERE/$TOKEN/" /opt/lichess-bot/config.yml
chmod 600 /opt/lichess-bot/config.yml

# Record exactly what is deployed (never hide what runs).
{ echo "engine tag:  $ENGINE_TAG ($(git -C /opt/sunfish rev-parse HEAD))"
  echo "net sha256:  $NET_SHA256"
  echo "bridge:      $LICHESS_BOT_COMMIT"
  echo "deployed:    $(date -u)"
} > /opt/sunfish/nnue_4k/DEPLOYED.txt

chown -R sunfish:sunfish /opt/lichess-bot /opt/sunfish

chmod +x /opt/sunfish/nnue_4k/lichess/watchdog.sh
cp /opt/sunfish/nnue_4k/lichess/sunfish-packed.service /etc/systemd/system/
cp /opt/sunfish/nnue_4k/lichess/sunfish-cost-gate.service /etc/systemd/system/
cp /opt/sunfish/nnue_4k/lichess/sunfish-watchdog.service /etc/systemd/system/
cp /opt/sunfish/nnue_4k/lichess/sunfish-watchdog.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now sunfish-cost-gate
systemctl enable --now sunfish-packed
systemctl enable --now sunfish-watchdog.timer

echo
echo "Done. Status:   systemctl status sunfish-packed"
echo "Logs:           journalctl -u sunfish-packed -f"
echo "Deployed build: cat /opt/sunfish/nnue_4k/DEPLOYED.txt"
echo "Cost tripwire:  systemctl status sunfish-cost-gate; cost_gate.py --status"
