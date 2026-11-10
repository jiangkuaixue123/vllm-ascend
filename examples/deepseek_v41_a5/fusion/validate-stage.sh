#!/usr/bin/env bash
set -euo pipefail
PIPELINE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEPLOY_DIR=$(cd -- "$PIPELINE_DIR/../.." && pwd)
cd "$DEPLOY_DIR"
mkdir -p logs
STAGE=${1:?Supply a unique candidate name}
if [ "${2:-all}" = all ]; then
  python3 source/examples/deepseek_v41_a5/verify-service.py --output "logs/fusion-${STAGE}-api.json" > "logs/fusion-${STAGE}-api.log" 2>&1
fi
docker exec -w /deployment jojo-v41-d4ac431 /opt/d4ac431/.venv/bin/python \
  source/tests/e2e/models/eval_v41_api.py \
  --config source/tests/e2e/models/configs/DeepSeek-V4.1-Flash-A5-FP8.yaml \
  --dataset reference/gsm8k-test.jsonl --output "/tmp/fusion-${STAGE}-gsm8k.json" \
  > "logs/fusion-${STAGE}-gsm8k.log" 2>&1
docker cp "jojo-v41-d4ac431:/tmp/fusion-${STAGE}-gsm8k.json" "logs/fusion-${STAGE}-gsm8k.json"
curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @reference/vision-request.json \
  > "logs/fusion-${STAGE}-vision.json"
python3 - "$STAGE" <<'PY'
import json, sys
from pathlib import Path
p = Path(f'logs/fusion-{sys.argv[1]}-vision.json')
result = json.loads(p.read_text())
assert result['choices'][0]['message']['content'].strip().rstrip('。.') == '5', result
PY
for RUN in 1 2 3; do
  python3 source/examples/deepseek_v41_a5/benchmark-fixed.py --output "logs/fusion-${STAGE}-throughput-${RUN}.json" \
    > "logs/fusion-${STAGE}-throughput-${RUN}.log" 2>&1
  python3 - "$STAGE" "$RUN" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(f'logs/fusion-{sys.argv[1]}-throughput-{sys.argv[2]}.json').read_text())
assert r['status'] == 'valid', r
print(sys.argv[1], sys.argv[2], r['status'], r['aggregate_output_tokens_per_second'], flush=True)
PY
done
