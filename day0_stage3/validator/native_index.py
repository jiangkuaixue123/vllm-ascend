# SPDX-License-Identifier: Apache-2.0
"""Explicit V4.1 paged index check. Requires PyTorch and NumPy. A5 also needs the installed native model package.

Run with --device cpu for the independent reference comparison. On a 9574, use
--device npu:0 --context-len 1048576 --output index.json for an actual capture
and replay through the ordinary installed Triton entry point.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

PAGE = 128
HEADS = 32
DIM = 128
BLOCK = 8
CHUNK = 1024
LEVELS = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def unpack(record):
    packed = record[..., :64].long()
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    lut = torch.tensor(LEVELS, device=record.device, dtype=torch.float32)
    scales = (record[..., 64:].int() << 23).contiguous().view(torch.float32)
    return (lut[codes].unflatten(-1, (4, 32)) * scales.unsqueeze(-1)).flatten(-2).bfloat16()


def records(count, generator):
    bank = torch.randint(0, 256, (count, 68), generator=generator, dtype=torch.uint8)
    # Fixed bounds make all 128 dot products and the sum of all 32 BF16 products
    # exactly representable in FP32 before each required BF16 rounding.
    bank[:, 64:] = torch.randint(126, 128, (count, 4), generator=generator, dtype=torch.uint8)
    # A unique four-byte physical row tag prevents periodic test fixtures or page aliases.
    for start in range(0, count, 65536):
        ids = torch.arange(start, min(start + 65536, count), dtype=torch.int32)
        bank[start : start + len(ids), :4] = torch.stack([(ids >> shift) & 255 for shift in (0, 8, 16, 24)], 1)
    return bank


def fixture(context, rows, ratio, seed, device):
    generator = torch.Generator().manual_seed(seed)
    pages_per_row = math.ceil(context / PAGE)
    pages = torch.randperm(pages_per_row * rows, generator=generator).reshape(rows, pages_per_row)
    if pages.unique().numel() != pages_per_row * rows:
        raise RuntimeError("Physical blocks alias between test requests")
    per_page = PAGE // ratio
    physical_rows = pages.numel() * per_page
    bank = records(physical_rows, generator)
    page_tags = bank[::per_page, :4].long()
    decoded_tags = (page_tags << torch.arange(0, 32, 8)).sum(1)
    if not torch.equal(decoded_tags, torch.arange(pages.numel()) * per_page):
        raise RuntimeError("Physical index pages do not carry independent row identities")
    # Query vectors use the same E2M1/UE8M0 lattice as the model, with independent payloads.
    qpack = torch.randint(0, 256, (rows, HEADS, 68), generator=generator, dtype=torch.uint8)
    qpack[..., 64:] = torch.randint(126, 128, (rows, HEADS, 4), generator=generator, dtype=torch.uint8)
    query = unpack(qpack)
    choices = torch.tensor([-7, -6, -5, -3, 3, 5, 6, 7], dtype=torch.float32) / 4
    weights = choices[torch.randint(len(choices), (rows, HEADS), generator=generator)].bfloat16()
    if rows > 1:
        # An entire row of equal scores exercises the device's tie order at k=512.
        weights[0].zero_()
    digest = hashlib.sha256()
    for tensor in (bank, pages, qpack, weights.view(torch.int16)):
        digest.update(memoryview(tensor.numpy()).cast("B"))
    return tuple(t.to(device) for t in (query, weights, bank, pages)), digest.hexdigest()


def candidate_ids(lengths, count, seed):
    # The model puts valid distinct candidate blocks first, in top-k score order,
    # forces the latest visible block, and represents all empty tail entries as -1.
    ids = torch.full((len(lengths), count), -1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(seed)
    for row, live in enumerate(lengths):
        available = math.ceil(live / BLOCK)
        if available:
            chosen = torch.randperm(available - 1, generator=generator)[: count - 1]
            ids[row, 0] = available - 1
            ids[row, 1 : 1 + len(chosen)] = chosen
    return ids


def stock_scores(inputs, lengths, blocks, width, ratio, max_live):
    """Original BF16 ATen operations, query batching and 1024-column BMM tiles."""
    q, weight, bank, pages = inputs
    result = q.new_full((len(q), width + 1), -torch.inf)
    scan = width if blocks is None else blocks.shape[1] * BLOCK
    # A wholly masked tile only writes -inf, which is already initialized. All
    # live BMMs keep the original shape, including the true last partial tile.
    reachable = max_live if blocks is None else min(math.ceil(max_live / BLOCK), blocks.shape[1]) * BLOCK
    for start in range(0, min(scan, reachable), CHUNK):
        seq = torch.arange(start, min(start + CHUNK, scan), device=q.device)
        if blocks is None:
            logical = seq.expand(len(q), -1)
        else:
            logical = blocks[:, seq // BLOCK].long() * BLOCK + seq % BLOCK
        visible = (logical >= 0) & (logical < lengths) & (logical < width)
        last = (logical.clamp(0, width - 1) + 1) * ratio - 1
        physical = (pages.gather(1, last // PAGE) * PAGE + last % PAGE) // ratio
        keys = unpack(bank[physical])
        dot = torch.bmm(q, keys.transpose(1, 2)).relu_()
        score = (dot * weight.unsqueeze(-1)).sum(1).masked_fill_(~visible, -torch.inf)
        if blocks is None:
            result[:, start : start + len(seq)].copy_(score)
        else:
            result.scatter_(1, torch.where(visible, logical, width), score)
    return result[:, :width].contiguous()


def independently_decode(packed):
    """Separate CPU FP64 key decoder for the direct mathematical reference."""
    code = torch.empty((len(packed), DIM), dtype=torch.int16)
    code[:, ::2] = packed[:, :64] % 16
    code[:, 1::2] = packed[:, :64] // 16
    mag = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), dtype=torch.float64)[code.long() % 8]
    signed = torch.where(code >= 8, -mag, mag)
    return signed * torch.exp2(packed[:, 64:].double() - 127).repeat_interleave(32, dim=1)


def direct_scores(inputs, logical_lengths, blocks, width, ratio):
    """Score each independent request in FP64 with the explicit BF16 round points."""
    q, weight, bank, pages = inputs
    output = torch.full((len(q), width), -torch.inf, dtype=torch.bfloat16)
    per_page = PAGE // ratio
    for row, live in enumerate(logical_lengths):
        columns = torch.arange(live)
        if blocks is not None:
            keep = torch.zeros(math.ceil(width / BLOCK), dtype=torch.bool)
            valid = blocks[row][blocks[row] >= 0].long()
            keep[valid] = True
            columns = columns[keep[columns // BLOCK]]
        for part in columns.split(2048):
            if not len(part):
                continue
            physical = pages[row, part // per_page] * per_page + part % per_page
            keys = independently_decode(bank[physical])
            dots = (q[row].double() @ keys.T).bfloat16().double().clamp_min_(0)
            products = (dots * weight[row].double().unsqueeze(-1)).bfloat16()
            output[row, part] = products.double().sum(0).bfloat16()
    return output


def topk(scores, lengths):
    raw = scores.topk(min(512, scores.size(1)), dim=-1, sorted=False).indices
    ordered = raw.sort(-1).values
    return raw, torch.where(ordered < lengths, ordered, -1)


def compare(actual, actual_ids, reference, lengths):
    # Always run both top-k operands on the selected device before moving the
    # results to the host. This includes the unsorted order and the model output.
    reference_ids = topk(reference, lengths)
    raw, ordered = [int((a.cpu() != b.cpu()).sum()) for a, b in zip(actual_ids, reference_ids)]
    a = actual.cpu().contiguous()
    b = reference.cpu().contiguous()
    wrong = a.view(torch.int16) != b.view(torch.int16)
    diff = int(wrong.sum())
    first = None
    if diff:
        row, col = divmod(int(wrong.int().flatten().argmax()), a.size(1))
        first = dict(
            row=row,
            column=col,
            observed_bits=int(a.view(torch.int16)[row, col]) & 65535,
            reference_bits=int(b.view(torch.int16)[row, col]) & 65535,
        )
    return dict(
        compared_scores=a.numel(),
        score_bit_differences=diff,
        nonfinite_mask_differences=int((a.isfinite() != b.isfinite()).sum()),
        first=first,
        unsorted_device_topk_differences=raw,
        model_device_topk_differences=ordered,
    )


def intermediate(device, seed, native):
    """Expose per-head dots, weighted products, and both intermediate roundings."""
    inputs, _ = fixture(512, 1, 1, seed, device)
    q, weight, bank, pages = inputs
    lengths = torch.full((1, 1), 512, dtype=torch.long, device=device)
    # Alternate the query sign so ReLU reveals both sides. A single one-hot head
    # makes the native score equal to that dot or that weighted dot.
    per_head = torch.eye(HEADS, dtype=torch.bfloat16, device=device)
    physical = (pages[0].cpu()[:, None] * PAGE + torch.arange(PAGE)).flatten()
    keys = independently_decode(bank.cpu()[physical])
    measured = {}
    dot_witnesses = product_witnesses = 0
    for sign in (1, -1):
        full = ((sign * q[0].cpu().double()) @ keys.T).clamp_min_(0)
        rounded = full.bfloat16().double()
        scaled = rounded * weight[0].cpu().double()[:, None]
        dot_witnesses += int(((full * weight[0].cpu().double()[:, None]).bfloat16() != scaled.bfloat16()).sum())
        product_witnesses += int((scaled.sum(0).bfloat16() != scaled.bfloat16().double().sum(0).bfloat16()).sum())
        qbatch = (q.expand(HEADS, -1, -1) * sign).contiguous()
        tables = pages.expand(HEADS, -1).contiguous()
        lens = lengths.expand(HEADS, -1).contiguous()
        for name, case, test_lengths, oracle in (
            ("dot", (qbatch, per_head, bank, tables), lens, rounded.bfloat16()),
            ("weighted_dot", (qbatch, per_head * weight, bank, tables), lens, scaled.bfloat16()),
            ("dense_sum", (q * sign, weight, bank, pages), lengths, scaled.bfloat16().double().sum(0).bfloat16()[None]),
        ):
            # A sum across zero-weight heads can canonicalize -0. Compare bits
            # against the actual stock reduction and numeric values against FP64.
            reference = stock_scores(case, test_lengths, None, 512, 1, 512)
            scores = native(*case, test_lengths, 512, 1, PAGE) if native else reference
            data = compare(scores, topk(scores, test_lengths), reference, test_lengths)
            data["independent_numeric_differences"] = int((scores.cpu() != oracle).sum())
            measured[f"{sign}_{name}"] = data
    if not dot_witnesses or not product_witnesses:
        raise RuntimeError("The supplied seed did not exercise both BF16 round points")
    return dict(
        compared_one_hot_heads=HEADS,
        keys=512,
        omitted_dot_rounding_witnesses=dot_witnesses,
        omitted_product_rounding_witnesses=product_witnesses,
        cases=measured,
    )


def trajectories(args):
    if args.lengths:
        states = [list(map(int, value.split(","))) for value in args.lengths]
    else:
        short = [min(args.context_len, 31 + 61 * i) for i in range(args.rows)]
        long = [(i + 1) * args.context_len // args.rows for i in range(args.rows)]
        shrink = [min(args.context_len, 2 + 13 * i) for i in range(args.rows)]
        shrink[-1] = 0
        if args.rows > 1:
            short[0] = 0
        states = [short, long, shrink]
    if len(states) < 2 or len({tuple(s) for s in states}) < 2:
        raise ValueError("Supply at least two different length vectors for replay")
    if any(len(s) != args.rows or any(n < 0 or n > args.context_len for n in s) for s in states):
        raise ValueError("Each --lengths must have one token count in [0, context_len] per row")
    for ratio in args.ratios:
        if len({tuple(n // ratio for n in state) for state in states}) < 2:
            raise ValueError(f"Provide at least two different compressed length vectors at ratio {ratio}")
    return states


def run_case(args, device, inputs, ratio, mode, states, native):
    width = max(1, args.context_len // ratio)
    count = min(args.candidate_blocks, math.ceil(width / BLOCK))
    lengths = torch.empty((args.rows, 1), dtype=torch.int64, device=device)
    blocks = torch.empty((args.rows, count), dtype=torch.int32, device=device) if mode == "candidate" else None
    graph = None
    result = []
    for step, tokens in enumerate(states):
        live = [value // ratio for value in tokens]
        lengths.copy_(torch.tensor(live, dtype=torch.long).reshape(-1, 1))
        if blocks is not None:
            blocks.copy_(candidate_ids(live, count, args.seed + 71 + step))
        ref = stock_scores(inputs, lengths, blocks, width, ratio, max(live))
        rec = dict(token_lengths=tokens, compressed_lengths=live)
        if native:
            if graph is None:
                scores = native(*inputs, lengths, width, ratio, PAGE, blocks)
                ids = topk(scores, lengths)
                torch.npu.synchronize(device)
                rec["ordinary_jit"] = compare(scores, ids, ref, lengths)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    replayed = native(*inputs, lengths, width, ratio, PAGE, blocks)
                    replay_ids = topk(replayed, lengths)
            graph.replay()
            torch.npu.synchronize(device)
            rec["same_graph_replay"] = compare(replayed, replay_ids, ref, lengths)
        else:
            observed = direct_scores(inputs, live, blocks, width, ratio)
            rec["cpu_independent"] = compare(observed, topk(observed, lengths), ref, lengths)
        result.append(rec)
    return dict(
        ratio=ratio,
        selection=mode,
        score_width=width,
        candidate_slots=count if blocks is not None else None,
        graph_objects=1 if native else 0,
        native_graph_replays=len(states) if native else 0,
        steps=result,
    )


def error_count(obj):
    if isinstance(obj, dict):
        return sum(value if key.endswith("differences") else error_count(value) for key, value in obj.items())
    if isinstance(obj, list):
        return sum(map(error_count, obj))
    return 0


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu or an explicit physical-device mapping such as npu:0")
    parser.add_argument("--context-len", type=int, default=4096, help="Logical tokens per request, up to 1048576")
    parser.add_argument("--rows", type=int, default=3, choices=range(1, 9))
    parser.add_argument("--seed", type=int, default=78829)
    parser.add_argument("--ratios", type=int, nargs="+", choices=(1, 2), default=(2, 1))
    parser.add_argument("--selection", choices=("both", "full", "candidate"), default="both")
    parser.add_argument("--candidate-blocks", type=int, default=2048)
    parser.add_argument("--lengths", action="append", help="Repeat with comma-separated token counts per request")
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 2 <= args.context_len <= 1048576 or args.candidate_blocks < 1 or args.cpu_threads < 1:
        parser.error("Expected 2..1048576 tokens, positive candidate-block and thread counts")
    states = trajectories(args)
    torch.set_num_threads(args.cpu_threads)
    native = None
    device_info = dict(type=args.device, torch=str(torch.__version__))
    if args.device != "cpu":
        if not args.device.startswith("npu:") or not args.device[4:].isdigit():
            parser.error("--device must be cpu or npu:<ordinal>")
        import torch_npu
        import triton
        from triton.runtime import driver

        from vllm_ascend.models.deepseek_v41.index_score_kernel import paged_index_scores

        device = torch.device(args.device)
        torch.npu.set_device(device)
        hw = torch.npu.get_device_name(device)
        arch = driver.active.get_current_target().arch
        if (hw, arch) != ("Ascend950DT_9574", "Ascend950DT_9574"):
            raise RuntimeError(f"Unexpected physical device or compiler target {(hw, arch)}")
        device_info.update(
            torch_npu=str(torch_npu.__version__), triton=str(triton.__version__), hardware=hw, jit_target=arch
        )
        native = paged_index_scores
    else:
        device = torch.device("cpu")
    report = dict(
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device=device_info,
        seed=args.seed,
        request_rows=args.rows,
        logical_context_tokens=args.context_len,
        backend="installed native index and ACLGraph" if native else "independent CPU references",
        intermediates=intermediate(device, args.seed, native),
        fixtures=[],
        cases=[],
    )
    modes = ("full", "candidate") if args.selection == "both" else (args.selection,)
    for ratio in args.ratios:
        inputs, digest = fixture(args.context_len, args.rows, ratio, args.seed + ratio, device)
        report["fixtures"].append(
            dict(
                ratio=ratio,
                sha256=digest,
                disjoint_physical_pages=inputs[-1].numel(),
                forced_tied_row=0 if args.rows > 1 else None,
                physical_rows=inputs[-2].size(0),
                physical_record_bytes=inputs[-2].numel(),
            )
        )
        for mode in modes:
            case = run_case(args, device, inputs, ratio, mode, states, native)
            report["cases"].append(case)
            print(f"ratio={ratio} selection={mode} compared {len(states)} length states", flush=True)
        del inputs
    errors = error_count(report)
    report["difference_count"] = errors
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if errors:
        raise SystemExit(f"Found {errors} comparison differences")


if __name__ == "__main__":
    main()
