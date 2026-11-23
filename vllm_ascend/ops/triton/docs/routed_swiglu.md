# Routed clipped SwiGLU

## Description

- **Function** fuses clipping and SwiGLU for the valid rows of a BF16 MC2 receive buffer.
- **Formula** computes `silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)`. The kernel rounds each clipped input to BF16 before applying SwiGLU.
- **Algorithm flow** reads the last cumulative count or sums per-expert counts to find the valid prefix. Vector cores process rows in that prefix. GMM2 consumes the same group list.
- **Supported modes** Ascend A3 eager execution and ACL Graph replay. The MoE caller uses the existing native activation when the Triton path is ineligible.

## Parameters

| Parameter | Role | Description | Data type | Layout |
| --- | --- | --- | --- | --- |
| `x` | Input | Packed gate and up output from GMM1 | BF16 | Contiguous `[capacity, 2 * width]` |
| `groups` | Input | Local expert token counts | INT32 or INT64 | Contiguous 1-D tensor on the device of `x` |
| `group_list_type` | Attribute | Cumulative counts (0) or per-expert counts (1) | Integer | N/A |
| `limit` | Attribute | Positive clipping limit | Float | N/A |
| Return value | Output | SwiGLU results for the valid prefix | BF16 | Contiguous `[capacity, width]` |

## Constraints

- Both packed halves have 32-byte row alignment. Each half is at most 4096 elements wide.
- The receive capacity and group list length are positive. The count reduction supports up to 64 local experts. Cumulative counts use a single load.
- The valid row count may change across graph replays and may be zero. The output tail is uninitialized, so callers must only consume rows described by the group list.
- Capacity is a runtime argument. Layout, group format and clipping limit are fixed for each compiled variant.

## Origin and Differences

Adapted from the [DeepSeek V4.1 Case 21 patch](https://github.com/jiaran-king/dsv4.1-optimize/blob/b786ab03d5a62e68e05d27292d4f1e1cb33c304a/patches/case21/ragged-swiglu/vllm-ascend.patch). The implementation uses the device vector-core count and integrates with the shared BF16 MoE MLP path.

## Test Cases

The tests compare valid outputs with separately clamped `torch_npu.npu_swiglu` at the DeepSeek V4.1 width. They check tail writes and compare GMM2 eager output with graph replay after the group counts change.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_routed_swiglu.py
```
