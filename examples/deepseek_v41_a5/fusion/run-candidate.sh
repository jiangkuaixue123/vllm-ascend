#!/usr/bin/env bash
# Run after candidate-specific numerical tests and an immutable image build.
set -euo pipefail
PIPELINE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEPLOY_DIR=$(cd -- "$PIPELINE_DIR/../.." && pwd)
PIPELINE_DIR=$DEPLOY_DIR/experiments/fusion-20261109
NAME=jojo-v41-d4ac431
STAGE=${1:?Supply a new stage name}
START_SCRIPT=$(realpath -- "${2:?Supply the prepared candidate start script}")
[[ "$STAGE" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo 'Invalid stage name' >&2; exit 1; }
[[ "$START_SCRIPT" == "$PIPELINE_DIR"/start-*.sh && -f "$START_SCRIPT" ]] || {
  echo 'Use a prepared start script in this experiment directory' >&2; exit 1;
}
bash -n "$START_SCRIPT"
cd "$DEPLOY_DIR"
mkdir -p logs
exec 9> "$PIPELINE_DIR/.pipeline.lock"
flock -n 9 || { echo 'Another candidate pipeline is running' >&2; exit 1; }
if compgen -G "logs/fusion-${STAGE}-*" >/dev/null; then
  echo 'Stage already has artifacts; choose a new name' >&2
  exit 1
fi
if pgrep -f '[b]enchmark-throughput.py|[b]enchmark-fixed.py|[v]alidate-stage.sh|[v]alidate-when-ready.sh' >/dev/null; then
  echo 'A benchmark or validation process is running' >&2
  exit 1
fi
OWNER=$(docker inspect -f '{{index .Config.Labels "owner"}}' "$NAME")
WORKLOAD=$(docker inspect -f '{{index .Config.Labels "workload"}}' "$NAME")
[[ "$OWNER" == jojo && "$WORKLOAD" == v41-d4ac431-a5-fp8-fp4 ]] || {
  echo 'Active container is not owned by this deployment' >&2; exit 1;
}
PREVIOUS=$NAME-before-$STAGE
FAILED=$NAME-failed-$STAGE
for RESERVED in "$PREVIOUS" "$FAILED"; do
  if docker inspect "$RESERVED" >/dev/null 2>&1; then
    echo "Reserved container name exists: $RESERVED" >&2
    exit 1
  fi
done
docker inspect "$NAME" > "logs/fusion-${STAGE}-previous-container.json"
docker logs "$NAME" > "logs/fusion-${STAGE}-previous-server.log" 2>&1
docker stop -t 60 "$NAME"
docker rename "$NAME" "$PREVIOUS"
if bash "$START_SCRIPT" > "logs/fusion-${STAGE}-start.log" 2>&1 &&
   bash "$PIPELINE_DIR/validate-when-ready.sh" "$STAGE" > "logs/fusion-${STAGE}-validation.log" 2>&1; then
  docker inspect "$NAME" > "logs/fusion-${STAGE}-container.json"
  docker logs "$NAME" > "logs/fusion-${STAGE}-server.log" 2>&1
  echo "Candidate validated; prior container retained as $PREVIOUS"
  echo 'Review results before updating the default start script and deployment state.'
  exit 0
fi
echo 'Candidate failed; restoring the previous container' >&2
if docker inspect "$NAME" >/dev/null 2>&1; then
  docker logs "$NAME" > "logs/fusion-${STAGE}-failed-server.log" 2>&1
  docker stop -t 60 "$NAME"
  docker rename "$NAME" "$FAILED"
fi
docker rename "$PREVIOUS" "$NAME"
docker start "$NAME"
echo 'Previous container restarted; wait for its health check before sending traffic.' >&2
exit 1
