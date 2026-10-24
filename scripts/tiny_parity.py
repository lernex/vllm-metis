# SPDX-License-Identifier: Apache-2.0
"""vLLM against the Hugging Face port on one checkpoint: prompt logprobs, decode logprobs, greedy.

The Hugging Face port (trust_remote_code, on CPU) is the reference. vLLM runs the same
checkpoint on the GPU through this plugin. Every prompt position is compared teacher-forced
(vLLM ``prompt_logprobs`` against the port's full forward), then the greedy continuations and
their per-step logprobs up to the first token where the two disagree.

    python scripts/tiny_parity.py --model /tmp/metis-tiny --dtype float32 --enforce-eager

Keep ``--gpu-memory-utilization`` small on a shared card; a tiny model needs very little.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

PROMPT_MESSAGES = [
    ([{"role": "user", "content": "What is the capital of France? Answer briefly."}], True),
    ([{"role": "system", "content": "You are terse."},
      {"role": "user", "content": "List three primes."}], False),
    ([{"role": "user", "content": "Write one sentence about the sea, then stop."}], True),
]
LONG_TEXT = (
    "Mamba layers carry a state from one chunk to the next, and attention layers keep every "
    "key and value. A long prompt crosses several 256-token chunks, so it checks both. "
)


def build_prompts(tokenizer, long_tokens: int) -> list[list[int]]:
    prompts = []
    for messages, thinking in PROMPT_MESSAGES:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=False
        )
        prompts.append(tokenizer.encode(text, add_special_tokens=False))
    ids: list[int] = []
    while len(ids) < long_tokens:
        ids += tokenizer.encode(LONG_TEXT, add_special_tokens=False)
    prompts.append(ids[:long_tokens])
    return prompts


FP8_MAX = 448.0


def _fake_fp8(x, dim=None):
    """Quantize to float8_e4m3 and back in FP32: one scale per tensor, or per row on ``dim``."""
    import torch

    xf = x.float()
    amax = xf.abs().amax() if dim is None else xf.abs().amax(dim=dim, keepdim=True)
    scale = torch.clamp(amax / FP8_MAX, min=1.0 / (FP8_MAX * 512.0))
    return (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale


def simulate_online_fp8(model) -> int:
    """Make the port compute what vLLM's online per-tensor fp8 computes, on the same layers.

    Linear layers (Mamba in/out projections, attention qkv/out): FP8 weights with one scale
    per tensor, activations quantized per token, FP32 accumulation. Routed experts (vLLM's
    Triton fp8 MoE): one weight scale per expert and matrix, the input and the SwiGLU output
    each quantized with one dynamic scale over the whole call, BF16 between the two GEMMs.
    Everything else stays as it is, as in vLLM.
    """
    import types

    import torch
    import torch.nn.functional as F

    def linear(lin):
        assert lin.bias is None
        weight = _fake_fp8(lin.weight.data)

        def forward(self, x):
            return F.linear(_fake_fp8(x, dim=-1), weight).to(x.dtype)

        lin.forward = types.MethodType(forward, lin)

    def experts(mod):
        gate_up = torch.stack([_fake_fp8(w) for w in mod.gate_up.weight.data])
        down = torch.stack([_fake_fp8(w) for w in mod.down.weight.data])

        def forward(self, hidden, indices, weights):
            tokens, k = indices.shape
            flat = indices.reshape(-1)
            token_of = torch.arange(tokens).repeat_interleave(k)
            quantized = _fake_fp8(hidden)
            inter = torch.empty(tokens * k, down.shape[-1], dtype=hidden.dtype)
            groups = [(int(e), (flat == e).nonzero().squeeze(-1)) for e in flat.unique()]
            for e, sel in groups:
                gate, up = F.linear(quantized[token_of[sel]], gate_up[e]).to(hidden.dtype).chunk(2, -1)
                inter[sel] = (F.silu(gate.float()) * up.float()).to(hidden.dtype)
            inter_q = _fake_fp8(inter)
            out = torch.zeros(tokens, hidden.shape[-1])
            routed_weights = weights.reshape(-1).float()
            for e, sel in groups:
                routed = F.linear(inter_q[sel], down[e]) * routed_weights[sel, None]
                out.index_add_(0, token_of[sel], routed.to(hidden.dtype).float())
            return out.to(hidden.dtype)

        mod.forward = types.MethodType(forward, mod)

    count = 0
    for layer in model.layers:
        if layer.is_attention:
            linear(layer.mixer.qkv)
            linear(layer.mixer.out)
        else:
            linear(layer.mixer.impl.mixer.in_proj)
            linear(layer.mixer.impl.mixer.out_proj)
        experts(layer.moe.local_experts)
        count += 3
    return count


def reference(model_dir, prompts, max_tokens, dtype_name, threads, fake_fp8=False):
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(threads)
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=dtype)
    model.eval()
    if fake_fp8:
        print(f"reference: simulated online fp8 on {simulate_online_fp8(model)} layers", flush=True)
    results = []
    with torch.no_grad():
        for ids in prompts:
            x = torch.tensor([ids])
            prompt_lp = torch.log_softmax(model(x, use_cache=False).logits[0].float(), -1)
            out = model.generate(
                x, max_new_tokens=max_tokens, do_sample=False, eos_token_id=None,
                pad_token_id=1, output_scores=True, return_dict_in_generate=True,
            )
            tokens = out.sequences[0, len(ids):].tolist()
            step_lp = [torch.log_softmax(s[0].float(), -1) for s in out.scores]
            results.append({"prompt_lp": prompt_lp, "tokens": tokens, "step_lp": step_lp})
    return results


def stop_engine(llm) -> None:
    """Shut the engine process down, so a fast exit cannot leave it holding the GPU."""
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # in-process engines have nothing to stop
        pass


def run_vllm(model_dir, prompts, max_tokens, args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.prefix_caching,
        quantization=args.quantization,
        seed=0,
        **({"compilation_config": {"cudagraph_capture_sizes": [1, 2, 4]}}
           if not args.enforce_eager else {}),
    )
    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, logprobs=args.topk,
        prompt_logprobs=args.topk, ignore_eos=True,
    )
    batches = {"batched": [TokensPrompt(prompt_token_ids=p) for p in prompts]}
    outputs = {}
    for name, batch in batches.items():
        outputs[name] = llm.generate(batch, params, use_tqdm=False)
    outputs["repeat"] = llm.generate(batches["batched"], params, use_tqdm=False)
    # One at a time, to see that batching does not change the answer.
    outputs["single"] = [llm.generate([b], params, use_tqdm=False)[0] for b in batches["batched"]]
    # Requests that ask for prompt logprobs bypass the prefix cache, so check cache hits with
    # decode logprobs only: the second round must reuse cached Mamba states and KV blocks.
    decode_only = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, logprobs=args.topk, ignore_eos=True
    )
    llm.generate(batches["batched"], decode_only, use_tqdm=False)
    outputs["cache_hit"] = llm.generate(batches["batched"], decode_only, use_tqdm=False)
    stop_engine(llm)
    return outputs


def compare(ref, out, prompt_ids):
    """Max |logprob difference| over the top-k vLLM reports, prompt and decode, plus greedy."""
    prompt_err = 0.0
    prompt_kl = 0.0
    positions = 0
    for t, entry in enumerate(out.prompt_logprobs or []):
        if entry is None:
            continue  # the first prompt token has no prediction
        hf = ref["prompt_lp"][t - 1]
        for token, lp in entry.items():
            prompt_err = max(prompt_err, abs(lp.logprob - hf[token].item()))
        prompt_kl += _topk_kl(entry, hf)
        positions += 1
    gen = out.outputs[0]
    tokens = list(gen.token_ids)
    agree = 0
    for a, b in zip(tokens, ref["tokens"]):
        if a != b:
            break
        agree += 1
    decode_err = 0.0
    # Step i's distribution is comparable while the contexts match: through the first mismatch.
    for i in range(min(agree + 1, len(tokens), len(ref["step_lp"]))):
        hf = ref["step_lp"][i]
        for token, lp in gen.logprobs[i].items():
            decode_err = max(decode_err, abs(lp.logprob - hf[token].item()))
    return {
        "prompt_tokens": len(prompt_ids),
        "cached_prompt_tokens": getattr(out, "num_cached_tokens", None),
        "prompt_max_abs_logprob_diff": prompt_err,
        "prompt_mean_topk_kl": prompt_kl / max(positions, 1),
        "greedy_agree": agree,
        "greedy_len": len(tokens),
        "decode_max_abs_logprob_diff_until_divergence": decode_err,
    }


def _topk_kl(entry, hf_lp) -> float:
    """KL(hf || vllm) over vLLM's reported tokens, each side renormalized on that support."""
    tokens = list(entry)
    v = [entry[t].logprob for t in tokens]
    h = [hf_lp[t].item() for t in tokens]
    lv = math.log(sum(math.exp(x) for x in v))
    lh = math.log(sum(math.exp(x) for x in h))
    return sum(math.exp(hh - lh) * ((hh - lh) - (vv - lv)) for hh, vv in zip(h, v))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--hf-dtype", default=None, help="reference dtype (default: --dtype)")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--quantization", default=None, help="e.g. fp8_per_tensor (online)")
    parser.add_argument("--hf-fake-fp8", action="store_true",
                        help="reference simulates vLLM's online per-tensor fp8")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.15)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--long-tokens", type=int, default=700)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--json", default=None, help="write the report here too")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = build_prompts(tokenizer, args.long_tokens)
    started = time.time()
    ref = reference(args.model, prompts, args.max_tokens, args.hf_dtype or args.dtype, args.threads,
                    fake_fp8=args.hf_fake_fp8)
    print(f"reference done in {time.time() - started:.1f}s", flush=True)
    outputs = run_vllm(args.model, prompts, args.max_tokens, args)
    report = {"config": vars(args), "runs": {}}
    for name, outs in outputs.items():
        report["runs"][name] = [compare(r, o, p) for r, o, p in zip(ref, outs, prompts)]
    singles = [list(o.outputs[0].token_ids) for o in outputs["single"]]
    report["batched_equals_single"] = [
        list(o.outputs[0].token_ids) == s for o, s in zip(outputs["batched"], singles)
    ]
    report["repeat_equals_first"] = [
        list(a.outputs[0].token_ids) == list(b.outputs[0].token_ids)
        for a, b in zip(outputs["batched"], outputs["repeat"])
    ]
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)
    os._exit(0)  # skip slow engine teardown


if __name__ == "__main__":
    main()
