# SPDX-License-Identifier: Apache-2.0
"""Every tensor vLLM holds after loading, against the checkpoint it came from.

Loads the model in vLLM, then, inside the worker, walks the checkpoint one tensor at a time
and finds where it landed: unquantized tensors must be bit-identical (FP32 ones still FP32);
online-fp8 tensors are dequantized and compared with the same quantization done here in
PyTorch. Also checks that every pass-1 Mamba slot runs on pass 0's modules after the quant
pass rewrote them. Cheap on memory: one checkpoint tensor at a time.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python scripts/check_loaded_weights.py --model DIR \\
        --quantization fp8_per_tensor --gpu-memory-utilization 0.15
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

FP8_MAX = 448.0


def check(model, checkpoint: str) -> dict:
    import torch
    from safetensors import safe_open

    params = dict(model.named_parameters(remove_duplicate=False))
    modules = dict(model.named_modules(remove_duplicate=False))
    report = {"exact": 0, "fp8": 0, "skipped": 0, "problems": []}

    def fp8_codes(t, scale):
        """The float8 values PyTorch picks for ``t / scale``, as floats."""
        return (t.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()

    def fp8_mismatch(stored, expected):
        """(values that differ, values more than one float8 step away) between float8 values."""
        magnitude = expected.abs().clamp_min(2.0**-6)
        step = torch.exp2(torch.floor(torch.log2(magnitude)) - 3)
        return (stored != expected).sum().item(), ((stored - expected).abs() > step * 1.0001).sum().item()

    def record_fp8(name, target, differ, too_far, total):
        # vLLM multiplies by 1/scale where this divides by scale, so a value sitting on a
        # rounding boundary may land one float8 step away; anything further is a bug.
        share = differ / total
        report["fp8"] += 1
        report["fp8_max_share_one_step_off"] = max(report.get("fp8_max_share_one_step_off", 0.0), share)
        if too_far or share > 0.01:
            report["problems"].append(
                f"{name} -> {target}: {share:.3%} differ, {too_far} by more than one step"
            )

    def checkpoint_tensors():
        for path in sorted(Path(checkpoint).glob("*.safetensors")):
            with safe_open(str(path), framework="pt") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    for name, tensor in checkpoint_tensors():
        mapped = model.map_checkpoint_name(name)
        if mapped is None:
            report["skipped"] += 1
            continue
        (target,) = mapped
        tensor = tensor.cuda()
        if target.endswith(".A") and name.endswith(".A_log"):
            tensor = -torch.exp(tensor.float())  # MambaMixer2 stores A = -exp(A_log)
        if target.endswith("w13_weight") or target.endswith("w2_weight"):
            layer = modules[target.rsplit(".", 1)[0]]
            stored = getattr(layer, target.rsplit(".", 1)[1])
            if stored.dtype == torch.float8_e4m3fn:
                scale = getattr(layer, target.rsplit(".", 1)[1] + "_scale").float().reshape(-1)
                # vLLM's per-expert scale is amax / 448 of exactly this expert's matrix.
                want = tensor.flatten(1).abs().amax(-1).float() / FP8_MAX
                if not torch.allclose(scale, want, rtol=1e-6):
                    report["problems"].append(f"{name}: expert scales are not amax/448")
                differ = too_far = 0
                for e in range(len(scale)):  # one expert at a time keeps the check small
                    d, t = fp8_mismatch(stored[e].float(), fp8_codes(tensor[e], scale[e]))
                    differ, too_far = differ + d, too_far + t
                record_fp8(name, target, differ, too_far, tensor.numel())
            else:
                if not torch.equal(stored, tensor.to(stored.dtype)):
                    report["problems"].append(f"{name} -> {target}: differs")
                report["exact"] += 1
            continue
        param = params[target]
        owner = modules[target.rsplit(".", 1)[0]]
        if param.dtype == torch.float8_e4m3fn:
            scale = owner.weight_scale.float().reshape(())
            # Online fp8 keeps the weight as [in, out] for the scaled-mm kernels, which may
            # pad the output dimension to a multiple of 16.
            stored = param.float().t()
            pad = stored[tensor.shape[0]:]
            if pad.numel() and pad.abs().max() != 0:
                report["problems"].append(f"{name}: kernel padding is not zero")
            stored = stored[: tensor.shape[0], : tensor.shape[1]]
            want = tensor.float().abs().amax() / FP8_MAX
            if not torch.allclose(scale, want, rtol=1e-6):
                report["problems"].append(f"{name}: scale {scale.item()} is not amax/448")
            record_fp8(name, target, *fp8_mismatch(stored, fp8_codes(tensor, scale)), tensor.numel())
            continue
        if param.dtype != tensor.dtype and tensor.dtype == torch.float32:
            report["problems"].append(f"{name}: FP32 in the checkpoint, {param.dtype} in vLLM")
        if not torch.equal(param.data, tensor.to(param.dtype)):
            report["problems"].append(f"{name} -> {target}: differs")
        report["exact"] += 1

    # The second pass must run on the first pass's (now quantized) modules.
    for i, layer in enumerate(model.model.layers):
        if layer.is_attention:
            continue
        first, second = layer.mixer.passes
        for part in ("in_proj", "conv1d", "out_proj", "norm"):
            if getattr(first, part) is not getattr(second, part):
                report["problems"].append(f"layer {i}: pass 1 has its own {part}")
        for part in ("A", "D", "dt_bias"):
            if getattr(first, part) is not getattr(second, part):
                report["problems"].append(f"layer {i}: pass 1 has its own {part}")
    return report


def stop_engine(llm) -> None:
    """Shut the engine process down, so a fast exit cannot leave it holding the GPU."""
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # in-process engines have nothing to stop
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--json", default=None, help="write the report here too")
    args = parser.parse_args()

    from vllm import LLM

    llm = LLM(
        model=args.model, trust_remote_code=True, dtype=args.dtype,
        quantization=args.quantization, gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=4, enforce_eager=True,
    )
    model_dir = args.model
    (report,) = llm.apply_model(lambda model: check(model, model_dir))
    print(json.dumps(report, indent=2), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    stop_engine(llm)
    os._exit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
