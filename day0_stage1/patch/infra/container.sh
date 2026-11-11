#!/usr/bin/env bash
# Run on the reserved A5 host with an existing device and port lease.
set -euo pipefail
: "${WORK:?absolute runtime work directory}" "${MODEL_PATH:?official checkpoint directory}"
: "${CONTAINER:?unique container name}" "${LEASED_DEVICES:?physical IDs from the coordinator}"
image=quay.io/ascend/vllm-ascend@sha256:73170914665aa65bf6f00b0d79dcb3b3a45fbda9af268a4763f5dec58e4393a9
[[ "$LEASED_DEVICES" == 0,1,2,3,4,5,6,7 ]]
devices=()
for node in /dev/davinci{0..7} /dev/davinci_manager /dev/hisi_hdc /dev/hisi_soc_cache_mgmt /dev/ubagg /dev/ubmempfd /dev/ubcore/* /dev/uburma/* /dev/ummu/* /dev/devmm_svm /dev/svm* /dev/umdk*; do
    [[ ! -c "$node" ]] || devices+=(--device "$node")
done
mounts=()
for p in /usr/local/Ascend/driver /usr/local/Ascend/add-ons /usr/local/dcmi /usr/local/bin/npu-smi /etc/ascend_install.info /etc/ascend_driver.conf /etc/ascend_ub_mode /etc/hccn.conf; do
    [[ ! -e "$p" ]] || mounts+=(-v "$p:$p:ro")
done
docker run -d --name "$CONTAINER" --cgroupns=host --security-opt label=disable --network bridge --shm-size=32g --ulimit memlock=-1:-1 \
    "${devices[@]}" "${mounts[@]}" \
    -v "$MODEL_PATH:/models/dsv41:ro" -v "$WORK:/stage1" \
    -e ASCEND_RT_VISIBLE_DEVICES="$LEASED_DEVICES" -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
    -e GLOO_SOCKET_IFNAME=lo -e OMP_NUM_THREADS=2 \
    "$image" sleep infinity
