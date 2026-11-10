#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NAME=jojo-v41-d4ac431
IMAGE=jojo/vllm-ascend:d4ac431-a5-fp8-fp4
COMMAND=(bash /deployment/serve.sh "$@")
if [ "${1:-}" = --debug-container ]; then
  COMMAND=(sleep infinity)
elif [ "${1:-}" = --eager ]; then
  shift
  COMMAND=(bash /deployment/serve-eager.sh "$@")
fi
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Container $NAME already exists; inspect it before restarting." >&2
  exit 1
fi
exec docker run -d --name "$NAME" \
  --restart unless-stopped \
  --log-opt max-size=100m --log-opt max-file=3 \
  --health-cmd 'curl --noproxy "*" -fsS http://127.0.0.1:8000/health || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 10m --health-retries 3 \
  --label owner=jojo --label workload=v41-d4ac431-a5-fp8-fp4 \
  --device /dev/davinci4 --device /dev/davinci5 \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/hisi_hdc \
  --device /dev/uburma --device /dev/ubcore --device /dev/ubmempfd \
  --device /dev/ubagg --device /dev/ummu \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --security-opt label=disable \
  --shm-size 64g -p 127.0.0.1:8000:8000 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/lib64/urma:/usr/lib64/urma:ro \
  -v /mnt/weight/DeepSeek-V4.1-Flash-1017:/models/deepseek-v41:ro \
  -v "$DEPLOY_DIR:/deployment:ro" \
  --entrypoint "${COMMAND[0]}" "$IMAGE" "${COMMAND[@]:1}"
