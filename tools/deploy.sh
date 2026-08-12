#!/bin/bash
# Production deploys. CI *tests* the deployed tree (tests/test_bot_integration.py
# plays a full game on the exact pin+patch+config bundle) but never deploys it;
# this script is the deploy.
#
#   tools/deploy.sh nnue    [user@host] [--check] [--sync-config]
#   tools/deploy.sh classic [user@host] [--check] [--sync-config]
#   tools/deploy.sh pypi vNNNN
#
# Both bots live on the Oracle A1 box (classic migrated off the GCP e2-micro
# 2026-08-12: burst-credit exhaustion cost it 2-3x nps during busy hours).
# They share the /opt/sunfish checkout but each has its own bridge install,
# selected by BOTDIR below.
#
# Bot deploy: update /opt/sunfish to origin/master (fast-forward only - a box
# with local commits aborts loudly), re-pin lichess-bot + re-apply the bundle
# patch if the pin moved, sync contrib files, smoke-test the engine BEFORE
# touching the running bot, wait for the account to be idle (lichess API,
# never journal greps), restart the unit, and verify the account comes back
# online. Config drift is reported, never silently overwritten: --sync-config
# rewrites the live config from the bundle template, preserving the token.
# --check reports all of the above and changes nothing.
#
# PyPI: verifies the tag matches pyproject's version and master's CI is green,
# then pushes the tag; the release workflow builds, smoke-tests the wheel,
# asserts the 4k byte budgets, and publishes via trusted publishing.
set -euo pipefail
cd "$(dirname "$0")/.."

die() { echo "deploy: $*" >&2; exit 1; }

MODE=${1:-}; shift || true
ACTION=deploy SYNC_CONFIG=0 HOST=""
for a in "$@"; do case $a in
    --check) ACTION=check ;;
    --sync-config) SYNC_CONFIG=1 ;;
    -*) die "unknown flag $a" ;;
    *) HOST=$a ;;
esac; done

case $MODE in
    nnue)    BUNDLE=nnue_4k/lichess UNIT=sunfish-packed  ACCOUNT=sunfish-nnue-engine
             BOTDIR=/opt/lichess-bot
             HOST=${HOST:-ubuntu@146.235.195.115} ;;
    classic) BUNDLE=tools/lichess   UNIT=sunfish-lichess ACCOUNT=sunfish-engine
             BOTDIR=/opt/lichess-bot-classic
             HOST=${HOST:-ubuntu@146.235.195.115} ;;
    pypi)
        TAG=${HOST:?usage: deploy.sh pypi vNNNN}
        VER=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
        [ "$TAG" = "v$VER" ] || die "tag $TAG does not match pyproject version v$VER - bump pyproject first"
        git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && die "tag $TAG already exists"
        git fetch -q origin master
        SHA=$(git rev-parse origin/master)
        CI=$(gh run list --commit "$SHA" --workflow python-app.yml --json conclusion -q '.[0].conclusion')
        [ "$CI" = "success" ] || die "CI on origin/master ($SHA) is '$CI', not success - not tagging"
        git tag -a "$TAG" -m "sunfish $VER" "$SHA"
        git push origin "$TAG"
        echo "Pushed $TAG at $SHA. The release workflow now builds, smoke-tests, asserts the"
        echo "4k budgets, and publishes to PyPI + a GitHub Release. Watch: gh run watch"
        exit 0 ;;
    *) die "usage: deploy.sh {nnue|classic|pypi} ..." ;;
esac

PIN=$(sed -n 's/^LICHESS_BOT_COMMIT=\([0-9a-f]*\)$/\1/p' "$BUNDLE/setup.sh")
ENGINE=$(sed -n 's/^  name: "\(.*\)"/\1/p' "$BUNDLE/config.yml")
[ -n "$PIN" ] && [ -n "$ENGINE" ] || die "could not parse pin/engine from $BUNDLE"

# The challenge-gate hook is per-box, not per-bundle: the Oracle box's gate is
# sunfish-cost-gate (flag /run/sunfish-costgate), which is what the nnue
# bundle's hook watches -- both bots deploy that one.  The classic bundle's
# hook is the GCP credit-gate variant and retired with the e2-micro.
HOOK_BUNDLE=nnue_4k/lichess

echo "== $MODE bot on $HOST: unit=$UNIT botdir=$BOTDIR engine=$ENGINE pin=${PIN:0:8} action=$ACTION"

playing() { curl -sf "https://lichess.org/api/users/status?ids=$ACCOUNT" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin)[0]; print("yes" if d.get("playing") else "no")'; }

# Phase 1 on the box: update, verify, sync, smoke - everything except the restart.
ssh "$HOST" "ACTION=$ACTION SYNC_CONFIG=$SYNC_CONFIG BUNDLE=$BUNDLE HOOK_BUNDLE=$HOOK_BUNDLE BOTDIR=$BOTDIR ENGINE=$ENGINE PIN=$PIN UNIT=$UNIT bash -s" <<'REMOTE'
set -euo pipefail
die() { echo "deploy[remote]: $*" >&2; exit 1; }
G() { sudo -u sunfish git -C /opt/sunfish "$@"; }

[ -d /opt/sunfish ] && [ -d "$BOTDIR" ] || die "/opt/sunfish or $BOTDIR missing - run setup.sh first"

# 1. Engine checkout -> origin/master, fast-forward only, loud otherwise.
G fetch -q origin master
# A locally-modified file is tolerated iff it is byte-identical to incoming
# master (a hotfix that has since been upstreamed); anything else aborts.
G status --porcelain --untracked-files=no | sed 's/^...//' | while IFS= read -r f; do
    G diff --quiet FETCH_HEAD -- "$f" \
        || die "modified tracked file in /opt/sunfish differs from origin/master - resolve by hand: $f"
    echo "note: $f is locally modified but matches origin/master (upstreamed hotfix) - will absorb"
done
# An untracked file that master now tracks blocks the checkout. Byte-identical
# copies (files deployed before they were committed) are absorbed; others abort.
G status --porcelain | sed -n 's/^?? //p' | while IFS= read -r f; do
    if G cat-file -e "FETCH_HEAD:$f" 2>/dev/null; then
        G cat-file blob "FETCH_HEAD:$f" | cmp -s - "/opt/sunfish/$f" \
            || die "untracked /opt/sunfish/$f differs from the now-tracked copy on origin/master - resolve by hand"
        [ "$ACTION" = check ] && echo "note: untracked $f matches origin/master (will become tracked)" || {
            sudo -u sunfish rm "/opt/sunfish/$f"; echo "absorbed untracked $f (identical, now tracked)"; }
    else echo "note: untracked $f (deploy artifact, e.g. net.sfnn - left alone)"; fi
done
OLD=$(G rev-parse HEAD); NEW=$(G rev-parse FETCH_HEAD)
if [ "$OLD" != "$NEW" ]; then
    G merge-base --is-ancestor HEAD FETCH_HEAD \
        || die "/opt/sunfish HEAD ($OLD) has commits not on origin/master - never discarded silently; inspect by hand"
    [ "$ACTION" = check ] && echo "would update: ${OLD:0:8} -> ${NEW:0:8}" || {
        G checkout -q --detach FETCH_HEAD; echo "engine updated: ${OLD:0:8} -> ${NEW:0:8}"; }
else echo "engine already at ${NEW:0:8}"; fi

# 2. lichess-bot pin + patch: the tested tree is the deployed tree.
LIVE_PIN=$(sudo -u sunfish git -C "$BOTDIR" rev-parse HEAD)
if [ "$LIVE_PIN" != "$PIN" ]; then
    echo "lichess-bot pin ${LIVE_PIN:0:8} != expected ${PIN:0:8}"
    [ "$ACTION" = check ] || {
        sudo -u sunfish git -C "$BOTDIR" fetch -q origin "$PIN"
        sudo -u sunfish git -C "$BOTDIR" checkout -qf "$PIN"
        sudo -u sunfish git -C "$BOTDIR" apply "/opt/sunfish/$BUNDLE/lichess-bot.patch"
        sudo "$BOTDIR/venv/bin/pip" install -q -r "$BOTDIR/requirements.txt"
        echo "re-pinned + patched + requirements refreshed"; }
else
    # Same pin: the PATCH may still have changed (it did on 2026-08-12, and
    # the fix silently didn't deploy). Compare patch-ids and re-apply.
    APPLIED=$(sudo -u sunfish git -C "$BOTDIR" diff | git patch-id --stable | cut -d" " -f1)
    WANT=$(sudo -u sunfish git -C /opt/sunfish show HEAD:"$BUNDLE/lichess-bot.patch" | git patch-id --stable | cut -d" " -f1)
    if [ -z "$APPLIED" ]; then
        echo "WARNING: lichess-bot tree is pristine - the bundle patch is NOT applied"
    elif [ "$APPLIED" != "$WANT" ]; then
        [ "$ACTION" = check ] && echo "would re-apply changed bundle patch" || {
            sudo -u sunfish git -C "$BOTDIR" checkout -qf "$PIN"
            sudo -u sunfish git -C "$BOTDIR" apply "/opt/sunfish/$BUNDLE/lichess-bot.patch"
            echo "re-applied changed bundle patch"; }
    fi
fi

# 3. Contrib files the bot reads from its install dir, not the repo.  The
# challenge-gate hook comes from HOOK_BUNDLE (the box's gate), not BUNDLE.
for f in extra_game_handlers.py; do
    if ! sudo cmp -s "/opt/sunfish/$HOOK_BUNDLE/$f" "$BOTDIR/$f"; then
        [ "$ACTION" = check ] && echo "would sync: $f" || {
            sudo cp "/opt/sunfish/$HOOK_BUNDLE/$f" "$BOTDIR/$f"; echo "synced: $f"; }
    fi
done

# 4. Config drift: report always, rewrite only on request, token preserved.
LIVE_CFG=$(sudo sed 's/^token:.*/token: MASKED/' "$BOTDIR/config.yml")
TPL_CFG=$(sed 's/^token:.*/token: MASKED/' "/opt/sunfish/$BUNDLE/config.yml")
DRIFT=$(diff <(printf '%s\n' "$LIVE_CFG") <(printf '%s\n' "$TPL_CFG")) || [ $? -eq 1 ] || die "config diff failed"
if [ -n "$DRIFT" ]; then
    echo "config drift (live vs bundle template):"; echo "$DRIFT"
    if [ "$SYNC_CONFIG" = 1 ] && [ "$ACTION" != check ]; then
        TOKEN=$(sudo sed -n 's/^token: *"\(.*\)"/\1/p' "$BOTDIR/config.yml")
        [ -n "$TOKEN" ] || die "could not extract live token - not rewriting config"
        # The token is fed via stdin, never argv: sudo logs every command
        # line to the journal, so a `sed s/.../$TOKEN/` would write the
        # secret into journalctl (observed live, 2026-08-12).
        printf '%s' "$TOKEN" | sudo env "TPL=/opt/sunfish/$BUNDLE/config.yml" python3 -c '
import os, sys
sys.stdout.write(open(os.environ["TPL"]).read().replace("YOUR_TOKEN_HERE", sys.stdin.read()))' \
            > /tmp/config.yml.new
        # -o/-g: the unit runs as sunfish; a root-owned 600 config is unreadable
        # to it (found live: took the bot down for 4 minutes on 2026-08-11)
        sudo install -m 600 -o sunfish -g sunfish /tmp/config.yml.new "$BOTDIR/config.yml"
        sudo rm /tmp/config.yml.new
        echo "config rewritten from template (token preserved)"
    else echo "(pass --sync-config to rewrite from the template; token is preserved)"; fi
fi

# 5. Engine smoke BEFORE the restart, under the unit's own Environment= (e.g.
# SF_NET): smoking a different net than the service uses proved worthless live
# (2026-08-11: smoke green, bot dead on a stale deployed net).
UNIT_ENV=$(systemctl show "$UNIT" -p Environment --value 2>/dev/null || true)
cd /opt/sunfish
if [ ! -f "$ENGINE" ] && [ "$ACTION" = check ]; then
    echo "engine $ENGINE not present at the current checkout - it arrives with the update"
else
    printf 'uci\nisready\nposition startpos\ngo movetime 200\nquit\n' \
        | timeout 60 sudo -u sunfish env $UNIT_ENV python3 "$ENGINE" | grep -q bestmove \
        || die "engine smoke failed: $ENGINE (env: ${UNIT_ENV:-none}) - if this is a stale
net, migrate it: python3 nnue_4k/packed/embed_tables.py NET_IN NET_OUT"
    echo "engine smoke ok: $ENGINE (env: ${UNIT_ENV:-none})"
fi
REMOTE

[ "$ACTION" = check ] && { echo "== check complete, nothing changed (account playing: $(playing))"; exit 0; }

# Phase 2: restart at idle - gate on the lichess API playing flag, never journal greps.
echo "waiting for $ACCOUNT to be idle..."
for i in $(seq 1 360); do
    [ "$(playing)" = no ] && break
    [ "$i" = 360 ] && die "$ACCOUNT still playing after 30 min - not restarting mid-game"
    sleep 5
done
ssh "$HOST" "sudo systemctl restart $UNIT && sleep 5 && sudo systemctl is-active --quiet $UNIT" \
    || { ssh "$HOST" "sudo journalctl -u $UNIT -n 30 --no-pager" >&2; die "$UNIT failed to come back - journal above"; }

echo "unit restarted; waiting for $ACCOUNT to appear online..."
for i in $(seq 1 36); do
    ONLINE=$(curl -sf "https://lichess.org/api/users/status?ids=$ACCOUNT" \
        | python3 -c 'import json,sys; print("yes" if json.load(sys.stdin)[0].get("online") else "no")')
    [ "$ONLINE" = yes ] && { echo "== $ACCOUNT online - deploy complete"; exit 0; }
    sleep 5
done
die "$ACCOUNT not online 3 min after restart - check: ssh $HOST journalctl -u $UNIT -f"
