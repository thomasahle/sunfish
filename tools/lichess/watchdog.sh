#!/bin/bash
# Liveness watchdog for the sunfish lichess bot -- the belt-and-braces layer
# behind the in-bridge fixes in lichess-bot.patch.  Production 2026-08-11: the
# bot sat connected-but-deaf for an hour while lichess started games against
# it; every failure mode of that shape ends the same way, with a move pending
# for us and nothing happening.  So that is what this watches: if lichess says
# it is our turn somewhere and the bot's journal has been silent for more than
# SILENCE_LIMIT seconds, the service is restarted (the bridge resumes live
# games on startup, so a restart is always safe).
#
# Run from a systemd timer (sunfish-watchdog.timer, every minute):
#   watchdog.sh <systemd-unit>       e.g. watchdog.sh sunfish-nnue
set -u

UNIT="${1:?usage: watchdog.sh <systemd-unit>}"
CONF=/opt/lichess-bot/config.yml
LOG=/var/log/sunfish-watchdog.log
SILENCE_LIMIT=90

TOK=$(grep -m1 -oE '^token: *"?[A-Za-z0-9_-]+' "$CONF" | sed 's/^token: *"\{0,1\}//')
[ -z "$TOK" ] && exit 0

# How many live games are waiting on OUR move.  A curl/API failure exits
# quietly: the watchdog must never restart the bot on its own bad network.
OUR_TURN=$(curl -sf -m 10 -H "Authorization: Bearer $TOK" \
        https://lichess.org/api/account/playing | python3 -c '
import json, sys
try:
    games = json.load(sys.stdin).get("nowPlaying", [])
except Exception:
    sys.exit(1)
print(sum(1 for g in games if g.get("isMyTurn")))' 2>/dev/null) || exit 0
[ -z "$OUR_TURN" ] && exit 0

if [ "$OUR_TURN" -gt 0 ]; then
    LAST=$(journalctl -u "$UNIT" -n 1 -o short-unix 2>/dev/null \
           | awk '{print $1}' | cut -d. -f1)
    NOW=$(date +%s)
    if [ -n "$LAST" ] && [ $((NOW - LAST)) -gt "$SILENCE_LIMIT" ]; then
        echo "$(date -u '+%F %T') watchdog: $OUR_TURN game(s) await our move" \
             "but $UNIT has been silent for $((NOW - LAST))s -- restarting" >> "$LOG"
        systemctl restart "$UNIT"
    fi
fi
