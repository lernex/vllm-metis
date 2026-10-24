# SPDX-License-Identifier: Apache-2.0
"""vLLM against the Transformers port on a tiny random Metis-1.6 checkpoint, on the GPU.

Each case runs ``scripts/tiny_parity.py`` in its own process (vLLM wants a fresh one per
engine) and holds the report to a bound. A few minutes per case, mostly torch.compile; the
engine takes 15% of the card.

    pytest -m gpu                         # METIS_TINY_DIR, or a tiny checkpoint is built from
                                          # METIS_MODEL_DIR's modeling files and tokenizer
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
# fp8 rounding makes the random tiny model chaotic: the CPU simulation alone moves by a top-20
# KL of about 7e-3 when its unquantized parts switch between BF16 and FP32. vLLM lands inside
# that (about 6e-3). A broken fp8 path (a wrong scale, a pass running unquantized weights as
# fp8) lands orders of magnitude further; test_loaded_weights is the exact check.
FP8_KL = 0.015
FP8_MAX_DIFF = 1.0
RELEASE = Path(os.environ.get("METIS_MODEL_DIR", Path.home() / "kronos-scratch/logos/hfport/stage"))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if os.environ.get("METIS_TINY_DIR"):
        return Path(os.environ["METIS_TINY_DIR"])
    if not (RELEASE / "modeling_metis16.py").exists():
        pytest.skip(f"no Metis release files at {RELEASE}")
    out = tmp_path_factory.mktemp("metis-tiny")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_tiny_checkpoint.py"),
         "--release-dir", str(RELEASE), "--out", str(out)],
        check=True,
    )
    return out


def small_fp32_moe_configs(folder: Path) -> Path:
    """Triton fused-MoE configs small enough for FP32 on cards with ~100 KB of shared memory.

    vLLM's default FP32 MoE tile overflows the shared memory of consumer Blackwell (SM120);
    BF16 and FP8, the serving dtypes, fit with the defaults.
    """
    import torch

    device = torch.cuda.get_device_name().replace(" ", "_")
    tile = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 2}
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"E=16,N=64,device_name={device},dtype=float32.json").write_text(
        json.dumps({str(m): tile for m in sizes})
    )
    return folder


def run(tiny: Path, tmp_path: Path, *args: str, env: dict | None = None) -> dict:
    report = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/tiny_parity.py"), "--model", str(tiny),
         "--json", str(report), *args],
        check=True, env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL,
    )
    return json.loads(report.read_text())


def worst(report: dict, key: str, runs=("batched", "repeat", "single")) -> float:
    return max(r[key] for name in runs for r in report["runs"][name])


def test_fp32_matches_port(tiny, tmp_path):
    """FP32 with IEEE dots (Triton's FP32 default is TF32): the same math, to rounding."""
    configs = small_fp32_moe_configs(tmp_path / "moe_configs")
    report = run(
        tiny, tmp_path, "--dtype", "float32", "--enforce-eager", "--prefix-caching",
        "--max-num-batched-tokens", "64",
        env={"TRITON_F32_DEFAULT": "ieee", "VLLM_TUNED_CONFIG_FOLDER": str(configs)},
    )
    assert worst(report, "prompt_max_abs_logprob_diff") < 1e-4
    assert worst(report, "decode_max_abs_logprob_diff_until_divergence") < 1e-4
    for name in ("batched", "repeat", "single", "cache_hit"):
        assert all(r["greedy_agree"] == r["greedy_len"] for r in report["runs"][name]), name
    assert report["runs"]["cache_hit"][-1]["cached_prompt_tokens"] > 0


def test_bf16_compiled(tiny, tmp_path):
    """The serving default: BF16, torch.compile, CUDA graphs, prefix caching."""
    report = run(tiny, tmp_path, "--dtype", "bfloat16", "--hf-dtype", "bfloat16", "--prefix-caching")
    assert worst(report, "prompt_mean_topk_kl") < 1e-3
    assert worst(report, "prompt_max_abs_logprob_diff") < 0.2
    assert worst(report, "decode_max_abs_logprob_diff_until_divergence", ("cache_hit",)) < 0.2
    assert all(report["repeat_equals_first"])
    assert report["runs"]["cache_hit"][-1]["cached_prompt_tokens"] > 0


def test_fp8_matches_simulated_fp8(tiny, tmp_path):
    """Online per-tensor fp8 against the port computing the same fp8 math on the CPU."""
    report = run(
        tiny, tmp_path, "--dtype", "bfloat16", "--quantization", "fp8_per_tensor",
        "--hf-fake-fp8", "--prefix-caching",
    )
    # Per-tensor activation scales span the whole batch in the MoE, so only lone requests
    # compute exactly what the simulation does.
    assert worst(report, "prompt_mean_topk_kl", ("single",)) < FP8_KL
    assert worst(report, "prompt_max_abs_logprob_diff", ("single",)) < FP8_MAX_DIFF
    assert all(report["repeat_equals_first"])


@pytest.mark.parametrize("quantization", [None, "fp8_per_tensor"])
def test_loaded_weights(tiny, tmp_path, quantization):
    """Every tensor vLLM holds is the checkpoint's, bit for bit, or its fp8 rounding."""
    args = ["--quantization", quantization] if quantization else []
    out = tmp_path / "weights.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_loaded_weights.py"), "--model", str(tiny),
         "--json", str(out), *args],
        env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}, stdout=subprocess.DEVNULL,
    )
    report = json.loads(out.read_text())
    assert result.returncode == 0 and not report["problems"], report["problems"]
    assert report["exact"] + report["fp8"] > 0
