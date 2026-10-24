# SPDX-License-Identifier: Apache-2.0
"""The vLLM model against the real checkpoint, without loading a byte of it.

Builds ``Metis16ForCausalLM`` for the release config on the meta device and checks every
tensor named in the release's safetensors header: where it maps, that the shapes and FP32
dtypes agree, that nothing in the model is left unloaded, and that the cache layers vLLM
will allocate are the 34 Mamba states and 6 KV caches the design calls for.

Set METIS_MODEL_DIR to the release directory (config.json and model.safetensors; only the
header of the weights file is read).
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from conftest import model_dir  # noqa: E402

MODEL_DIR = model_dir(need_weights=True)
if MODEL_DIR is None or not (MODEL_DIR / "model.safetensors").exists():
    pytest.skip("no release checkpoint: set METIS_MODEL_DIR", allow_module_level=True)
WEIGHTS = MODEL_DIR / "model.safetensors"

import torch  # noqa: E402

import vllm_metis  # noqa: E402

vllm_metis.register()


def safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


@pytest.fixture(scope="module")
def built():
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.engine.arg_utils import EngineArgs

    from vllm_metis.model import Metis16ForCausalLM

    vllm_config = EngineArgs(
        model=str(MODEL_DIR), trust_remote_code=True, max_model_len=4096, load_format="dummy"
    ).create_engine_config()
    with set_current_vllm_config(vllm_config):
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.init_distributed_environment(
                world_size=1, rank=0, local_rank=0,
                distributed_init_method="tcp://127.0.0.1:29517", backend="gloo",
            )
            parallel_state.initialize_model_parallel(1, 1)
    with set_current_vllm_config(vllm_config), torch.device("meta"):
        torch.set_default_dtype(vllm_config.model_config.dtype)
        model = Metis16ForCausalLM(vllm_config=vllm_config)
        torch.set_default_dtype(torch.float32)
    return vllm_config, model


def test_every_checkpoint_tensor_maps(built):
    _, model = built
    header = safetensors_header(WEIGHTS)
    params = dict(model.named_parameters())
    covered: dict[str, str] = {}
    skipped = []
    for name, info in header.items():
        mapped = model.map_checkpoint_name(name)
        if mapped is None:
            skipped.append(name)
            continue
        (target,) = mapped
        assert target in params, f"{name} -> {target} is not a model parameter"
        assert target not in covered, f"{name} and {covered.get(target)} both map to {target}"
        covered[target] = name
        param = params[target]
        assert list(param.shape) == info["shape"], (name, list(param.shape), info["shape"])
        if info["dtype"] == "F32":
            assert param.dtype == torch.float32, f"{target} must stay FP32"
        else:
            assert param.dtype == torch.bfloat16, f"{target} is {param.dtype}"
    unloaded = sorted(set(params) - set(covered))
    assert not unloaded, f"model parameters with no checkpoint tensor: {unloaded}"
    # What is skipped: the tied head and the adaptive heads the fixed policy never reads.
    expected_skips = {"lm_head.weight", "depth_budget.multiplier"}
    for i in range(20):
        expected_skips |= {
            f"layers.{i}.moe.k_router.weight",
            f"layers.{i}.moe.k_router.bias",
            f"layers.{i}.moe.k_budget.multiplier",
        }
    assert set(skipped) == expected_skips
    loaded_elements = sum(
        int(torch.Size(info["shape"]).numel()) for n, info in header.items() if n not in skipped
    )
    model_elements = sum(p.numel() for p in params.values())
    assert loaded_elements == model_elements
    assert abs(model_elements / 1e9 - 10.23) < 0.01, model_elements


def test_cache_layers(built):
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.models.utils import extract_layer_index

    from vllm_metis.model import MetisMambaMixer

    vllm_config, model = built
    context = vllm_config.compilation_config.static_forward_context
    attention = sorted(n for n, m in context.items() if isinstance(m, Attention))
    mamba = sorted(n for n, m in context.items() if isinstance(m, MetisMambaMixer))
    assert len(attention) == 6 and len(mamba) == 34
    assert {extract_layer_index(n) for n in attention} == {5, 10, 15}
    for name in attention + mamba:
        extract_layer_index(name)  # vLLM requires exactly one integer in every cache layer name
    # Every pass-1 Mamba slot shares all of its parameters with pass 0.
    for i, layer in enumerate(model.model.layers):
        if layer.is_attention:
            continue
        first, second = layer.mixer.passes
        assert second.prefix != first.prefix
        for (n1, p1), (n2, p2) in zip(
            first.named_parameters(remove_duplicate=False),
            second.named_parameters(remove_duplicate=False),
        ):
            assert n1 == n2 and p1 is p2, (i, n1)
        assert second.conv_weights is first.conv_weights
    shapes = type(model).get_mamba_state_shape_from_config(vllm_config)
    for name in mamba:
        assert tuple(context[name].get_state_shape()) == tuple(shapes)
    assert type(model).get_mamba_state_dtype_from_config(vllm_config)[1] == torch.float32


def test_ssm_state_defaults_to_fp32(built):
    vllm_config, _ = built
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == "float32"
    assert vllm_config.model_config.is_hybrid
