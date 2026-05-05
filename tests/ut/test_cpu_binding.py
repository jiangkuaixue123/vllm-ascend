from unittest.mock import MagicMock, patch

from vllm_ascend.cpu_binding import bind_cpus


def _cpu_info():
    return {numa: list(range(numa * 40, (numa + 1) * 40)) for numa in range(8)}


def _device_map_info():
    return {device: MagicMock() for device in range(16)}


@patch("vllm_ascend.cpu_binding._get_cpu_info", return_value=_cpu_info())
@patch("vllm_ascend.cpu_binding._get_numa_node_count", return_value=8)
@patch("vllm_ascend.cpu_binding._get_device_map_info",
       return_value=_device_map_info())
@patch("vllm_ascend.cpu_binding.psutil.Process")
@patch.dict("os.environ", {"ASCEND_RT_VISIBLE_DEVICES": "11"}, clear=False)
def test_bind_cpus_uses_global_device_mapping_for_single_visible_device(
        mock_process, *_):
    process = MagicMock()
    process.cpu_affinity.side_effect = [None, list(range(220, 240))]
    mock_process.return_value = process

    bind_cpus(rank_id=0, ratio=1.0)

    process.cpu_affinity.assert_any_call(list(range(220, 240)))


@patch("vllm_ascend.cpu_binding._get_cpu_info", return_value=_cpu_info())
@patch("vllm_ascend.cpu_binding._get_numa_node_count", return_value=8)
@patch("vllm_ascend.cpu_binding._get_device_map_info",
       return_value=_device_map_info())
@patch("vllm_ascend.cpu_binding.psutil.Process")
@patch.dict("os.environ", {
    "ASCEND_RT_VISIBLE_DEVICES": "11",
    "CPU_BINDING_NUM": "10",
}, clear=False)
def test_bind_cpus_honors_cpu_binding_num_with_global_mapping(
        mock_process, *_):
    process = MagicMock()
    process.cpu_affinity.side_effect = [None, list(range(210, 220))]
    mock_process.return_value = process

    bind_cpus(rank_id=0, ratio=1.0)

    process.cpu_affinity.assert_any_call(list(range(210, 220)))


@patch("vllm_ascend.cpu_binding._get_cpu_info", return_value=_cpu_info())
@patch("vllm_ascend.cpu_binding._get_numa_node_count", return_value=8)
@patch("vllm_ascend.cpu_binding._get_device_map_info")
@patch("vllm_ascend.cpu_binding.psutil.Process")
@patch.dict("os.environ", {
    "ASCEND_RT_VISIBLE_DEVICES": "11",
    "CPU_BINDING_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
}, clear=False)
def test_bind_cpus_allows_explicit_global_device_list(mock_process,
                                                      mock_device_map, *_):
    process = MagicMock()
    process.cpu_affinity.side_effect = [None, list(range(220, 240))]
    mock_process.return_value = process

    bind_cpus(rank_id=0, ratio=1.0)

    mock_device_map.assert_not_called()
    process.cpu_affinity.assert_any_call(list(range(220, 240)))
