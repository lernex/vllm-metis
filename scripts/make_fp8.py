# SPDX-License-Identifier: Apache-2.0
"""Write an FP8 copy of a Metis-1.6 checkpoint for vLLM and SGLang.

    python scripts/make_fp8.py <bf16 checkpoint dir> <output dir>

The output is a compressed-tensors ``FP8_DYNAMIC`` checkpoint: FP8 (E4M3) weights with one
scale per output channel, activations quantized per token at run time. vLLM (with
``vllm-metis``) and SGLang (with ``sglang-metis``) load it directly, so a model that is too
large to hold in BF16 still fits: Logos is 11 GB instead of 20 GB.

Only the large matrices the engines quantize are converted: the Mamba input projection, the
attention projections and the routed experts. Everything the architecture is sensitive to
stays BF16 or FP32 exactly as trained: the Mamba output projection, the MoE latent
projections, the shared expert, the routers, the depth memory and the embeddings. Transformers
does not read this format for this architecture; use the BF16 checkpoint there.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
SHARD_BYTES = 5 * 1024**3
QUANTIZED_SUFFIXES = (
    ".mixer.impl.mixer.in_proj.weight",
    ".mixer.qkv.weight",
    ".mixer.out.weight",
    ".moe.local_experts.gate_up.weight",
    ".moe.local_experts.down.weight",
)
QUANTIZATION_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8, "type": "float", "strategy": "channel",
                "symmetric": True, "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8, "type": "float", "strategy": "token",
                "symmetric": True, "dynamic": True,
            },
            "output_activations": None,
        }
    },
    # The Mamba output projection stays BF16 (attention's projection is named ``out``).
    "ignore": ["lm_head", "re:.*out_proj$"],
    "kv_cache_scheme": None,
}


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel FP8: rows of the last two dims, each stacked expert on its own."""
    wide = weight.float()
    scale = (wide.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
    return (wide / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.safetensors"))
    shard: dict[str, torch.Tensor] = {}
    size, shards, weight_map = 0, [], {}

    def flush() -> None:
        nonlocal shard, size
        if shard:
            name = f"model-{len(shards) + 1:05d}.safetensors"
            save_file(shard, target / name, metadata={"format": "pt"})
            shards.append((name, list(shard)))
            shard, size = {}, 0

    quantized = 0
    for path in files:
        with safe_open(path, "pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key.endswith(QUANTIZED_SUFFIXES):
                    tensor, scale = quantize(tensor)
                    out = {key: tensor, key + "_scale": scale}
                    quantized += 1
                else:
                    out = {key: tensor}
                for k, v in out.items():
                    shard[k] = v.contiguous()
                    size += v.numel() * v.element_size()
                if size >= SHARD_BYTES:
                    flush()
    flush()

    total = len(shards)
    for i, (name, keys) in enumerate(shards, 1):
        final = f"model-{i:05d}-of-{total:05d}.safetensors"
        (target / name).rename(target / final)
        weight_map.update({k: final for k in keys})
    total_size = sum((target / n).stat().st_size for n in set(weight_map.values()))
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
    )

    for item in source.iterdir():
        if item.suffix == ".safetensors" or item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item.resolve(), dest, dirs_exist_ok=True)
        else:
            shutil.copyfile(item.resolve(), dest)
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = QUANTIZATION_CONFIG
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"quantized {quantized} tensors into {total} shards, {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
