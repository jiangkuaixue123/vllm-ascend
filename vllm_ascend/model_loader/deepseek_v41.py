# SPDX-License-Identifier: Apache-2.0
"""Load original V4.1 safetensors through the standard vLLM loader registry."""
import json
import os
from functools import cache
from pathlib import Path

import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    maybe_download_from_modelscope,
)

from vllm_ascend import envs
from vllm_ascend.models.deepseek_v41.weights import load_checkpoint


@cache
def register_deepseek_v41_loader():
    """Register once per process. Called by NPUPlatform before creating the loader."""
    register_model_loader("deepseek_v41")(DeepseekV41ModelLoader)


class DeepseekV41ModelLoader(BaseModelLoader):
    def load_model(self, vllm_config, model_config, prefix=""):
        model = super().load_model(vllm_config, model_config, prefix)
        finalize = getattr(model, "finalize_after_load", None)
        if finalize is not None:
            with set_current_vllm_config(vllm_config, prefix=prefix):
                finalize(receipt_dir=envs.VLLM_ASCEND_DSV41_RESULTS_DIR)
            model.eval()
        return model

    def _checkpoint_dir(self, model_config):
        name = model_config.model
        revision = model_config.revision
        # Users may supply either a model directory or a revision/pinned snapshot
        # in the Hub cache. The vLLM helpers also honor HF_HUB_OFFLINE and its lock.
        local = Path(name).expanduser()
        if not local.is_dir():
            local = Path(maybe_download_from_modelscope(name, revision) or name)
        if not local.is_dir():
            cache_dir = self.load_config.download_dir
            local = Path(download_weights_from_hf(
                name, cache_dir, ["*.safetensors"], revision,
                ignore_patterns=self.load_config.ignore_patterns,
            ))
            if not (local / "model.safetensors.index.json").is_file() and not (local / "model.safetensors").is_file():
                download_safetensors_index_file_from_hf(
                    name, "model.safetensors.index.json", cache_dir=cache_dir, revision=revision,
                )
        if not any((local / f).is_file() for f in ("model.safetensors.index.json", "model.safetensors")):
            raise FileNotFoundError(f"Could not find V4.1 safetensors or their index in {local}")
        return local

    def download_model(self, model_config):
        self._checkpoint_dir(model_config)

    def load_weights(self, model, model_config):
        root = self._checkpoint_dir(model_config)
        hf = model_config.hf_config
        quant = getattr(hf, "checkpoint_quantization", None) or getattr(hf, "quantization_config", None)
        if quant is None or (
            quant.get("quant_method"), quant.get("activation_scheme"), quant.get("weight_block_size"),
            quant.get("scale_fmt"), quant.get("expert_dtype")
        ) != ("fp8", "dynamic", [32, 32], "ue8m0", "fp4"):
            raise ValueError("Expected dynamic V4.1 FP8/UE8M0 block scales and packed FP4 experts")
        backbone = model.backbone
        ep = backbone.placement
        device = backbone.device
        output = envs.VLLM_ASCEND_DSV41_RESULTS_DIR
        manifest = None
        if output:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            manifest = output / f"rank-{ep.rank}-load.json"
        load_checkpoint(backbone, root, ep_rank=ep.rank, ep_size=ep.size, device=device,
                        manifest=manifest, hf_config=hf)
        if output:
            props = dict(ep_rank=ep.rank, ep_size=ep.size, dp_rank=model.dp_rank,
                         tp_size=1, pp_size=1, pcp_size=1, model_layers=len(backbone.layers),
                         expert_range=model.expert_range, worker_pid=os.getpid())
            if hasattr(os, "sched_getaffinity"):
                props["cpu_affinity"] = sorted(os.sched_getaffinity(0))
            if torch.device(device).type == "npu":
                props.update(device=torch.npu.current_device(), npu_name=torch.npu.get_device_name(device),
                             memory_free_total=torch.npu.mem_get_info(device),
                             peak_torch_allocated=torch.npu.max_memory_allocated(device))
            with (output / f"rank-{ep.rank}-placement.json").open("w") as stream:
                json.dump(props, stream)
