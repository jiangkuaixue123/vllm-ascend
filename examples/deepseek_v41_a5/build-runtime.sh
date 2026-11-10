#!/usr/bin/env bash
set -euo pipefail

# Run inside the isolated jojo-v41-d4ac431-build container.
install -d /opt/d4ac431
test -d /opt/d4ac431/vllm || cp -r /deployment/vllm /opt/d4ac431/vllm
test -d /opt/d4ac431/vllm-ascend || cp -r /deployment/source /opt/d4ac431/vllm-ascend
test "$(git -C /opt/d4ac431/vllm rev-parse HEAD)" = 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
git -C /opt/d4ac431/vllm-ascend merge-base --is-ancestor d4ac4319acdaf480689c39b6320a998712f80523 HEAD

if ! command -v uv >/dev/null; then
  curl -LsSf --max-time 120 https://astral.sh/uv/install.sh -o /tmp/d4ac431-install-uv.sh
  UV_INSTALL_DIR=/usr/local/bin sh /tmp/d4ac431-install-uv.sh
fi
cd /opt/d4ac431
test -x .venv/bin/python || uv venv --system-site-packages --python "$(command -v python)" .venv
export VIRTUAL_ENV=/opt/d4ac431/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH=/opt/d4ac431/vllm:/opt/d4ac431/vllm-ascend
export VLLM_TARGET_DEVICE=empty
export SOC_VERSION=ascend950dt_9582
export MAX_JOBS=16
export CARGO_BUILD_JOBS=8
export MAKEFLAGS=-j16
export CMAKE_BUILD_PARALLEL_LEVEL=16

# Build this deployment's CANN operator only; other models' kernels are unused.
cp /deployment/source/csrc/build_aclnn.sh /opt/d4ac431/vllm-ascend/csrc/build_aclnn.sh
trap 'cp /deployment/source/csrc/build_aclnn.sh /opt/d4ac431/vllm-ascend/csrc/build_aclnn.sh' EXIT
python - <<'PY'
from pathlib import Path
p = Path('/opt/d4ac431/vllm-ascend/csrc/build_aclnn.sh')
text = p.read_text()
start = text.index('    CUSTOM_OPS_ARRAY=(', text.index('elif [[ "$SOC_VERSION" =~ ^ascend950'))
end = text.index('    )', start) + len('    )')
p.write_text(text[:start] + '    CUSTOM_OPS_ARRAY=("sparse_flash_attention")' + text[end:])
PY

uv pip install --python .venv/bin/python --no-deps \
  'torch-npu==2.10.0.post4' 'triton-ascend==3.2.2' \
  --index-url https://pypi.org/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e /opt/d4ac431/vllm
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e /opt/d4ac431/vllm-ascend
.venv/bin/python -c 'import importlib.metadata as m; print({p: m.version(p) for p in ["torch", "torch-npu", "triton-ascend", "vllm", "vllm-ascend", "transformers"]})'
.venv/bin/python -c 'from vllm.plugins import load_general_plugins; load_general_plugins(); from vllm_ascend.models.deepseek_v41 import model; print(model.__file__)'
