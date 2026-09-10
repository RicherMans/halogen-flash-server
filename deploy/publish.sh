#!/usr/bin/env bash
# deploy/publish.sh: build and push the fork image.
#
# The version tag comes from the Dockerfile base image, so bumping
# deploy/Dockerfile (as deploy/sync-upstream.yml does) is the single source of
# truth. Each build is pushed under that version AND under :latest.
#
# Intended to run on GitHub Actions where GITHUB_REPOSITORY and a logged-in
# container registry are present. IMAGE_NS overrides the namespace for local
# testing; PUBLISH_DISABLED=1 builds without pushing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 deploy/llama-swap-bridge.py --selftest

VER="$(sed -n 's/^FROM ghcr.io\/peonist-ai\/halogen-flash-server://p' Dockerfile)"
[ -n "$VER" ] || { echo "publish: no FROM version found in Dockerfile" >&2; exit 1; }

NS="${IMAGE_NS:-ghcr.io/${GITHUB_REPOSITORY,,}}"
IMAGE="$NS:$VER"

docker build -f Dockerfile -t "$IMAGE" .
echo "publish: built $IMAGE"

if [ "${PUBLISH_DISABLED:-0}" != "1" ]; then
  docker push "$IMAGE"
  docker tag "$IMAGE" "$NS:latest"
  docker push "$NS:latest"
  echo "publish: pushed $IMAGE and $NS:latest"
else
  echo "publish: PUBLISH_DISABLED=1, not pushing"
fi