"""Eager expert dispatch for independently scheduled DP requests on one EP group."""
import torch
import torch.distributed as dist


class ExpertPlacement:
    def __init__(self, group):
        self.group = group
        if group is None:
            if dist.is_initialized() and dist.get_world_size() != 1:
                raise ValueError("An explicit EP device process group is required")
            self.size, self.rank = 1, 0
        else:
            self.size = dist.get_world_size(group)
            self.rank = dist.get_rank(group)

    def token_width(self, tokens, device):
        """Establish one padded width for all layers in this step (including idle DP)."""
        if self.size == 1:
            return tokens
        largest = torch.tensor([tokens], dtype=torch.int32, device=device)
        dist.all_reduce(largest, op=dist.ReduceOp.MAX, group=self.group)
        return int(largest.item())

    def gather(self, tensor, width, fill):
        if self.size == 1:
            return tensor
        padded = tensor.new_full((width, tensor.size(1)), fill)
        if tensor.size(0):
            padded[:tensor.size(0)].copy_(tensor)
        output = padded.new_empty(self.size * width, padded.size(1))
        dist.all_gather_into_tensor(output, padded, group=self.group)
        return output

    def routed(self, experts, start_id, end_id, local_x, local_indices, local_weights, width):
        if width == 0:
            return local_x.float()
        all_x = self.gather(local_x, width, 0)
        all_ids = self.gather(local_indices.int(), width, -1)
        all_weights = self.gather(local_weights, width, 0)
        output = torch.zeros_like(all_x, dtype=torch.float32)

        # Group assignments once on the host. This removes per-expert device
        # synchronizations and excludes padding from computation completely.
        ids = all_ids.cpu()
        rows, slots = torch.where((ids >= start_id) & (ids < end_id))
        if len(rows):
            keys = ids[rows, slots]
            order = keys.argsort(stable=True)
            expert_ids, sizes = keys[order].unique_consecutive(return_counts=True)
            rows, slots = rows[order].to(all_x.device), slots[order].to(all_x.device)
            offset = 0
            for expert_id, size in zip(expert_ids.tolist(), sizes.tolist()):
                selected = rows[offset:offset + size]
                mix = all_weights[selected, slots[offset:offset + size], None]
                contribution = experts[expert_id](all_x[selected], mix).float()
                # Each top-k includes an expert at most once per token.
                output[selected] = output[selected] + contribution
                offset += size
        if self.size > 1:
            dist.all_reduce(output, group=self.group)
        begin = self.rank * width
        return output[begin:begin + local_x.size(0)]
