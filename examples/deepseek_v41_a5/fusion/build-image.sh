#!/usr/bin/env bash
# Build from committed source; this command does not access an NPU or start a service.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
IMAGE=${1:-jojo/vllm-ascend:dsv41-triton-fusion}
BASE=d4ac4319acdaf480689c39b6320a998712f80523
git -C "$REPO_DIR" merge-base --is-ancestor "$BASE" HEAD
if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
  echo 'Commit or preserve local edits before building an immutable source image.' >&2
  exit 1
fi
SOURCE_COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD)
BUILD_DIR=$(mktemp -d)
trap 'rm -rf -- "$BUILD_DIR"' EXIT
git -C "$REPO_DIR" bundle create "$BUILD_DIR/source.bundle" HEAD "^$BASE"
cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/Dockerfile"
docker build --build-arg "SOURCE_COMMIT=$SOURCE_COMMIT" -t "$IMAGE" "$BUILD_DIR"
