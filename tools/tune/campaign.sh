#!/bin/sh
set -eu

: "${CAMPAIGN_LABEL:?set CAMPAIGN_LABEL}"
: "${CAMPAIGN_REGISTRY:?set CAMPAIGN_REGISTRY}"
: "${CAMPAIGN_HEARTBEAT:?set CAMPAIGN_HEARTBEAT}"
: "${CAMPAIGN_CHECKPOINT:?set CAMPAIGN_CHECKPOINT}"
: "${CAMPAIGN_TERM_GRACE:=30}"
: "${CAMPAIGN_KILL_GRACE:=5}"
[ "$#" -gt 0 ] || { echo "usage: campaign.sh COMMAND [ARG ...]" >&2; exit 2; }
command -v setsid >/dev/null || { echo "campaign.sh requires setsid" >&2; exit 2; }

owner=$(id -u)
mkdir -p "$(dirname "$CAMPAIGN_REGISTRY")" "$(dirname "$CAMPAIGN_HEARTBEAT")"

checkpoint() {
    if [ -e "$CAMPAIGN_CHECKPOINT" ]; then
        stat -c '%s:%Y' "$CAMPAIGN_CHECKPOINT"
    else
        printf 'missing'
    fi
}

record() {
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    line="$now\t$CAMPAIGN_LABEL\t$pid\t$pgid\t$1\t${2:-0}\t$(checkpoint)"
    printf '%b\n' "$line" >> "$CAMPAIGN_REGISTRY"
    printf '%b\n' "$line" > "$CAMPAIGN_HEARTBEAT.tmp"
    mv "$CAMPAIGN_HEARTBEAT.tmp" "$CAMPAIGN_HEARTBEAT"
}

other_cpu() {
    ps -eo uid=,pcpu= | awk -v owner="$owner" \
        '$1 >= 1000 && $1 < 65534 && $1 != owner { total += $2 } END { print total + 0 }'
}

group_alive() {
    ps -eo pgid=,stat= | awk -v pgid="$pgid" \
        '$1 == pgid && $2 !~ /^Z/ { live = 1 } END { exit !live }'
}

stop_group() {
    record "$1" "$2"
    kill -TERM -- "-$pgid" 2>/dev/null || true
    remaining=$CAMPAIGN_TERM_GRACE
    while group_alive && [ "$remaining" -gt 0 ]; do
        sleep 1
        remaining=$((remaining - 1))
    done
    if group_alive; then
        record term-survivors "$2"
        kill -KILL -- "-$pgid" 2>/dev/null || true
        remaining=$CAMPAIGN_KILL_GRACE
        while group_alive && [ "$remaining" -gt 0 ]; do
            sleep 1
            remaining=$((remaining - 1))
        done
        if group_alive; then
            record kill-survivors "$2"
            return 1
        fi
        wait "$pid" 2>/dev/null || true
        record killed "$2"
        return
    fi
    wait "$pid" 2>/dev/null || true
    record stopped "$2"
}

setsid "$@" &
pid=$!
pgid=
remaining=40
while [ "$remaining" -gt 0 ]; do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$pid" != "$pgid" ] || break
    sleep .05
    remaining=$((remaining - 1))
done
if [ "$pid" != "$pgid" ]; then
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo "campaign controller has no private process group" >&2
    exit 1
fi
trap 'stop_group external-signal 0; exit 143' HUP INT TERM
record launched 0

while kill -0 "$pid" 2>/dev/null; do
    first=$(other_cpu)
    sleep 5
    second=$(other_cpu)
    record heartbeat "$second"
    if awk -v a="$first" -v b="$second" 'BEGIN { exit !(a > 100 && b > 100) }'; then
        stop_group preempt "$second"
        exit 75
    fi
done

set +e
wait "$pid"
status=$?
set -e
record "exit-$status" 0
exit "$status"
