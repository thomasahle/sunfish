#!/bin/bash
# Sets up the sunfish lichess bot on a fresh Debian/Ubuntu VM.
# Usage (as root, on the VM):
#   curl -sL https://raw.githubusercontent.com/thomasahle/sunfish/master/contrib/lichess/setup.sh \
#     | sudo bash -s -- <LICHESS_BOT_TOKEN>
set -euo pipefail

TOKEN="${1:-${LICHESS_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
    echo "Usage: setup.sh <LICHESS_BOT_TOKEN>" >&2
    echo "Create one at https://lichess.org/account/oauth/token/create?scopes[]=bot:play" >&2
    echo "while logged in to the bot account." >&2
    exit 1
fi

apt-get update -qq
apt-get install -y -qq git pypy3 python3-venv python3-pip

# 1GB of swap keeps a 1GB free-tier VM comfortable
if [ ! -f /swapfile ]; then
    fallocate -l 1G /swapfile && chmod 600 /swapfile
    mkswap /swapfile && swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

id -u sunfish &>/dev/null || useradd -r -m sunfish

[ -d /opt/sunfish ] || git clone --depth 1 https://github.com/thomasahle/sunfish /opt/sunfish
[ -d /opt/lichess-bot ] || git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot /opt/lichess-bot

python3 -m venv /opt/lichess-bot/venv
/opt/lichess-bot/venv/bin/pip install -q -r /opt/lichess-bot/requirements.txt

cp /opt/sunfish/contrib/lichess/config.yml /opt/lichess-bot/config.yml
sed -i "s/YOUR_TOKEN_HERE/$TOKEN/" /opt/lichess-bot/config.yml
chmod 600 /opt/lichess-bot/config.yml

# The CPU-credit gate. lichess-bot calls is_supported_extra() from this file on
# every incoming challenge; it declines while /run/sunfish-throttled exists.
# This replaces the no-op stub that ships with lichess-bot.
cp /opt/sunfish/contrib/lichess/extra_game_handlers.py /opt/lichess-bot/

# User *and* group: a later `git pull` runs as sunfish and must be able to
# write .git/objects, .git/index and .git/HEAD. Running git as root inside
# /opt/sunfish leaves root-owned objects that break the next unprivileged
# pull; repair with `chown -R sunfish:sunfish /opt/sunfish`.
chown -R sunfish:sunfish /opt/lichess-bot /opt/sunfish

cp /opt/sunfish/contrib/lichess/sunfish-lichess.service /etc/systemd/system/
cp /opt/sunfish/contrib/lichess/sunfish-credit-gate.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now sunfish-credit-gate
systemctl enable --now sunfish-lichess

echo
echo "Done. Check status with:  systemctl status sunfish-lichess"
echo "Follow the logs with:     journalctl -u sunfish-lichess -f"
echo "CPU-credit gate:          systemctl status sunfish-credit-gate"
echo "Gate snapshot:            python3 /opt/sunfish/contrib/lichess/cpu_credit.py --status"
