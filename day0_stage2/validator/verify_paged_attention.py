# SPDX-License-Identifier: Apache-2.0
"""Compare the paged attention layers with the in-repository serial eager implementation."""

import argparse
import json
from copy import deepcopy
from types import SimpleNamespace

import torch

from vllm_ascend.models.deepseek_v41.backend import Attention, SharedAttentionRuntime, set_dtype
from vllm_ascend.models.deepseek_v41.decode_state import DecodeInputs, PagedHistory
from vllm_ascend.models.deepseek_v41.graph_ops import LayerPages, paged_attention


def serial_reference(layers, vectors, prompts):
    """Store each prefix's serial output when its real prefill or decode executes."""
    rows = {}

    def forward(stack, prefix, tokens):
        start = len(prefix)
        if start and len(tokens) != 1:
            raise ValueError("HF continuation consumes one token per call")
        x = vectors.index_select(0, torch.tensor(tokens, device=vectors.device)).unsqueeze(0)
        keys = [prefix + tokens[: i + 1] for i in range(len(tokens))]
        for key in keys:
            if key in rows:
                raise ValueError("Reference prefix evaluated twice")
            rows[key] = []
        for attn in stack:
            output = attn(x, start)[0].float().cpu()
            selected = [None] * len(tokens)
            if attn.is_index_source:
                # HF concatenates the entire prompt KV on prefill, and the
                # fixed-width KV ring on decode, in front of compressed keys.
                offset = attn.window_size if start else len(tokens)
                selected = [[j - offset for j in indices if j >= 0] for indices in attn.shared.topk_idxs[0].tolist()]
                assert all(j >= 0 for indices in selected for j in indices)
            for key, value, indices in zip(keys, output, selected, strict=True):
                rows[key].append((value, indices))
        return keys[-1]

    def decode(stack, prefix, tokens):
        for token in tokens:
            prefix = forward(stack, prefix, (token,))
        return prefix

    # Each request owns its ring, compression state and shared index buffers.
    # Slice the one full prompt output by prefix when observing chunked paged prefill.
    for req, prompt in prompts.items():
        stack = deepcopy(layers)
        prefix = forward(stack, (), prompt)
        if req == "a":
            prefix = decode(stack, prefix, (7,))
            # Fork the committed state before evaluating the discarded verify suffix.
            decode(deepcopy(stack), prefix, (8, 9, 10, 11, 12))
            decode(stack, prefix, (17, 18))
        else:
            decode(stack, prefix, range(26, 32) if req == "b" else (39, 40, 41))
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--capture", action="store_true")
    opts = parser.parse_args()
    dev = torch.device(opts.device)
    if dev.type == "npu":
        torch.npu.set_device(dev)
    torch.set_num_threads(2)
    torch.manual_seed(804)
    cfg = SimpleNamespace(
        dim=128,
        n_heads=2,
        q_lora_rank=64,
        o_lora_rank=32,
        head_dim=64,
        rope_head_dim=32,
        o_groups=2,
        window_size=4,
        compress_ratios=[0, 2, 2, 1, 1, 1],
        norm_eps=1e-6,
        n_layers=6,
        kv_source_layers=[1, 3],
        index_source_layers=[1, 3, 4],
        candidate_source_layer=3,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        index_n_heads=2,
        index_head_dim=64,
        index_topk=2,
        original_seq_len=0,
        rope_theta=10000,
        compress_rope_theta=10000,
        rope_factor=1,
        beta_fast=32,
        beta_slow=1,
        max_batch_size=1,
        max_seq_len=16,
    )

    def layers():
        shared = SharedAttentionRuntime()
        with torch.device(dev), set_dtype(torch.bfloat16):
            return [Attention(i, cfg, shared) for i in range(cfg.n_layers)]

    source = layers()
    for attn in source:
        for name, p in attn.named_parameters():
            if p.dtype == torch.uint8:
                p.fill_(124)
            elif name.endswith("norm.weight"):
                p.fill_(1)
            else:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * 0.09).to(dtype=p.dtype, device=dev))
    history = PagedHistory(cfg.max_seq_len, 4, 12, max_draft_tokens=5)
    stage = DecodeInputs(history, 16, dev, [])
    banks = [LayerPages(a, stage) for a in source]
    physical = {"a": [2, 5, 7, 1], "b": [3, 6, 10, 11], "c": [2, 5, 7, 1]}
    prompts = {"a": tuple(range(1, 7)), "b": tuple(range(21, 26)), "c": tuple(range(35, 39))}
    # x is deliberately a per-token deterministic input. All six layers
    # receive the same sequence of vectors so shared KV/index behavior stays comparable.
    vectors = torch.randn(60, cfg.dim, dtype=torch.float32).bfloat16().to(dev)
    reference = serial_reference(source, vectors, prompts)

    def stage_rows(positions, slots, requests, ends):
        plan = history.plan(
            positions, slots, [physical[r] for r in requests], ends, prompt_lengths=[len(prompts[r]) for r in requests]
        )
        stage.stage_layout(plan)

    chains = {k: [] for k in physical}
    observations = []
    for tag, chunks, padding in [
        ("interleaved-prefill", [("a", 0, [1, 2, 3]), ("b", 0, [21, 22])], 1),
        ("chunked-prefill-reordered", [("b", 2, [23, 24, 25]), ("a", 3, [4, 5, 6])], 0),
        ("verify-six", [("a", 6, [7, 8, 9, 10, 11, 12]), ("b", 5, [26])], 0),
        ("reject-and-overwrite", [("b", 6, [27]), ("a", 7, [17, 18])], 2),
        ("recycle-page-and-continue-peer", [("c", 0, [35, 36, 37, 38]), ("b", 7, [28])], 0),
    ]:
        source_ids = []
        positions = []
        slots = []
        ends = []
        requests = []
        expected = []
        for key, start, add in chunks:
            seq = chains[key][:start] + add
            chains[key] = seq
            table = physical[key]
            for pos, token in enumerate(add, start):
                expected.append(reference[tuple(seq[: pos + 1])])
                source_ids.append(token)
                positions.append(pos)
                slots.append(table[pos // 4] * 4 + pos % 4)
            ends.append(len(source_ids))
            requests.append(key)
        count = len(source_ids)
        positions += [0] * padding
        source_ids += [0] * padding
        slots += [-1] * padding
        stage_rows(positions, slots, requests, ends)
        shared = {}
        x = vectors.index_select(0, torch.tensor(source_ids, device=dev)).unsqueeze(0)
        before = []
        for i, a in enumerate(source):
            out = paged_attention(a, x, stage, banks[i], shared, max(positions) + 1)
            target = torch.stack([row[i][0] for row in expected])
            real = out[0, :count].float().cpu()
            diff = (target - real).abs()
            if not torch.isfinite(real).all():
                raise AssertionError(f"nonfinite layer {i} at {tag}")
            route_diff = []
            if a.is_index_source:
                current = shared["selected"][:count].cpu().tolist()
                for t, (row, expected_row) in enumerate(zip(current, expected, strict=True)):
                    golden = expected_row[i][1]
                    chosen = [k for k in row if k >= 0]
                    if chosen != golden:
                        route_diff.append(dict(token=t, eager=golden, paged=chosen))
            before.append(
                dict(
                    layer=i,
                    route_differences=route_diff,
                    max_abs=float(diff.max()),
                    rel_l2=float(diff.norm() / target.norm()),
                    unequal=int((diff != 0).sum()),
                    values=int(diff.numel()),
                )
            )
        observations.append(dict(schedule=tag, layers=before))
    replay_deltas = []
    if opts.capture and dev.type != "npu":
        raise ValueError("Capture is an NPU measurement")

    def fixed_shape_replays():
        ids = torch.tensor([29, 39], device=dev)

        def run_graph():
            shared = {}
            x = vectors.index_select(0, ids).unsqueeze(0)
            return torch.stack(
                [paged_attention(a, x, stage, b, shared, history.max_len) for a, b in zip(source, banks)]
            )

        def prepare(reqs, values):
            p = [len(chains[r]) for r in reqs]
            slots = [physical[r][pos // 4] * 4 + pos % 4 for r, pos in zip(reqs, p)]
            stage_rows(p, slots, reqs, [1, 2])
            ids.copy_(torch.tensor(values))

        if opts.capture:
            prepare(["b", "c"], [29, 39])
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream(device=dev)
            with torch.npu.stream(stream):
                run_graph()
            stream.synchronize()
            with torch.npu.graph(graph):
                actual = run_graph()
        for reqs, values in ((["b", "c"], [29, 39]), (["c", "b"], [40, 30]), (["b", "c"], [31, 41])):
            prepare(reqs, values)
            if opts.capture:
                graph.replay()
                torch.npu.synchronize()
            else:
                actual = run_graph()
            eager = []
            for req, token in zip(reqs, values):
                chains[req].append(token)
                eager.append(torch.stack([out for out, _ in reference[tuple(chains[req])]]))
            delta = (actual[:, 0].float().cpu() - torch.stack(eager, dim=1)).abs()
            replay_deltas.append(
                dict(requests=reqs, max_abs=float(delta.max()), unequal=int((delta != 0).sum()), values=delta.numel())
            )

    fixed_shape_replays()
    print(
        json.dumps(
            dict(
                device=str(dev),
                scope=(
                    "6 reduced-dimension paged attention layers versus one fixed prefill then serial decode; "
                    + ("NPU graph replay" if opts.capture else "eager fixed-shape replay")
                ),
                config=dict(ratios=cfg.compress_ratios, source=cfg.kv_source_layers, topk=cfg.index_topk),
                observations=observations,
                graph_replays=replay_deltas,
            )
        )
    )


if __name__ == "__main__":
    main()
