#!/usr/bin/env bash
set -euo pipefail
PIPELINE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEPLOY_DIR=$(cd -- "$PIPELINE_DIR/../.." && pwd)
cd "$DEPLOY_DIR"
STAGE=${1:?Supply a unique candidate name}
for ATTEMPT in $(seq 1 180); do
  if curl --noproxy '*' -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    exec bash experiments/fusion-20261109/validate-stage.sh "$STAGE"
  fi
  sleep 5
done
echo 'Service did not become healthy within 15 minutes' >&2
exit 1
