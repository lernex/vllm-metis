# SPDX-License-Identifier: Apache-2.0
"""Write a tiny random Metis-1.6 checkpoint laid out like the release.

Same module names, the same FP32 tensors (routers, balance bias, SSM parameters, depth head),
the release tokenizer and chat template, and shapes vLLM's kernels accept (head_dim 64,
Mamba chunk 256). Weights are random but pushed away from their identity-like init, so every
path is live: the memory gates open, the per-pass adapters and balance bias are non-zero.

    python scripts/make_tiny_checkpoint.py --release-dir <Metis-1.6-Logos dir> --out /tmp/metis-tiny
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import torch

COPY = (
    "configuration_metis16.py",
    "modeling_metis16.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)

TINY = dict(
    vocab_size=65536,
    hidden_size=256,
    num_hidden_layers=6,
    attention_layer_indices=[1, 4],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    attention_pass_lora_rank=8,
    mamba_d_state=32,
    mamba_d_conv=4,
    mamba_expand=2,
    mamba_head_dim=64,
    mamba_n_groups=2,
    mamba_chunk_size=256,
    mhc_pass_embedding_dim=16,
    latent_dim=128,
    n_routed_experts=16,
    num_experts_per_tok=4,
    max_routed_k=8,
    expert_intermediate_size=64,
    shared_expert_intermediate_size=64,
    route_feature_dim=32,
    memory_dim=32,
    max_position_embeddings=8192,
)


def load_port(release: Path):
    """Import the release's own configuration and modeling files as a package."""
    package = "metis_release_port"
    spec = importlib.util.spec_from_file_location(
        package, release / "modeling_metis16.py", submodule_search_locations=[str(release)]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[package] = pkg
    for name in ("configuration_metis16", "modeling_metis16"):
        sub_spec = importlib.util.spec_from_file_location(f"{package}.{name}", release / f"{name}.py")
        module = importlib.util.module_from_spec(sub_spec)
        sys.modules[f"{package}.{name}"] = module
        sub_spec.loader.exec_module(module)
    return sys.modules[f"{package}.configuration_metis16"], sys.modules[f"{package}.modeling_metis16"]


def build(release: Path, out: Path, seed: int = 0, **overrides) -> Path:
    configuration, modeling = load_port(release)
    torch.manual_seed(seed)
    config = configuration.Metis16Config(**{**TINY, **overrides})
    config.auto_map = {
        "AutoConfig": "configuration_metis16.Metis16Config",
        "AutoModelForCausalLM": "modeling_metis16.Metis16ForCausalLM",
    }
    config.architectures = ["Metis16ForCausalLM"]
    model = modeling.Metis16ForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 1 and "norm" not in name and "A_log" not in name:
                p.add_(torch.randn_like(p) * 0.05)
        for layer in model.layers:
            layer.moe.selection_bias.normal_(0.0, 0.5)
            if layer.is_attention:
                layer.mixer.pass_gates.normal_(0.0, 1.0)
                layer.mixer.pass_lora_up.weight.normal_(0.0, 0.05)
        # Open the memory gates (the init keeps them nearly shut) so the memory path matters.
        model.depth_memory.stream_gate_bias.normal_(0.0, 0.5)
        model.depth_memory.stream_gate.normal_(0.0, 0.5)
    model = model.to(torch.bfloat16)
    # What the release holds in FP32.
    for layer in model.layers:
        layer.moe.expert_router.float()
        layer.moe.k_router.float()
        layer.moe.selection_bias.data = layer.moe.selection_bias.data.float()
        layer.moe.k_budget.multiplier.data = layer.moe.k_budget.multiplier.data.float()
        if not layer.is_attention:
            core = layer.mixer.impl.mixer
            for name in ("A_log", "D", "dt_bias"):
                getattr(core, name).data = getattr(core, name).data.float()
    model.continuation.float()
    model.depth_budget.multiplier.data = model.depth_budget.multiplier.data.float()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    for name in COPY:
        if (release / name).exists():
            shutil.copyfile(release / name, out / name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(build(args.release_dir, args.out, args.seed))


if __name__ == "__main__":
    main()
