#!/bin/bash
# Creates a free-tier GCP e2-micro VM and installs the sunfish lichess bot.
# Prerequisites: `gcloud auth login` done, a project selected
# (`gcloud config set project <id>`) with billing enabled.
# Usage: ./gcp-create.sh <LICHESS_BOT_TOKEN>
set -euo pipefail

TOKEN="${1:?Usage: gcp-create.sh <LICHESS_BOT_TOKEN>}"
# Free tier requires us-west1, us-central1 or us-east1 + e2-micro + <=30GB standard disk
ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-sunfish-lichess}"

gcloud compute instances create "$NAME" \
    --zone="$ZONE" \
    --machine-type=e2-micro \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-standard

echo "Waiting for the VM to boot..."
sleep 30

gcloud compute ssh "$NAME" --zone="$ZONE" --command \
    "curl -sL https://raw.githubusercontent.com/thomasahle/sunfish/master/tools/lichess/setup.sh | sudo bash -s -- '$TOKEN'"

echo
echo "sunfish is live. Watch it: gcloud compute ssh $NAME --zone=$ZONE --command 'journalctl -u sunfish-lichess -f'"
