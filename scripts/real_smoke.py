# SPDX-License-Identifier: Apache-2.0
"""Metis-1.6 Logos in vLLM with its real weights: logit parity, generation, parsers, speed.

1. Teacher-forced parity. vLLM's top-k prompt logprobs on fixed token sequences against two
   references computed on CPU in FP32 by the release tooling: the native training runtime
   (``native_logits.pt``) and the Transformers port (``port_<name>.pt``). The port's own
   distance to native is printed beside vLLM's, so the gap can be read against it.
2. Greedy generation in thinking and direct mode, and a tool prompt in each mode, decoded by
   the same composed parser ``--reasoning-parser metis --tool-call-parser metis`` builds.
3. Throughput: one stream, then a batch.

Needs a free GPU. Online fp8 (the default here) holds the weights in about 10.6 GB; bf16
needs about 21 GB for the weights alone (``--quantization none``).

    python scripts/real_smoke.py --model <Metis-1.6-Logos dir> --refs <dir with prompts.json,
        native_logits.pt, port_*.pt> --json real_fp8.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "days": {"type": "integer", "description": "Forecast length in days"},
                },
                "required": ["city"],
            },
        },
    }
]
CHAT = [{"role": "user", "content": "What is 17 x 23? Answer briefly."}]
TOOL_CHAT = [{"role": "user", "content": "What's the weather in Lisbon for the next 3 days?"}]


def topk_kl(entry, ref_lp) -> float:
    """KL(ref || vllm) on vLLM's reported tokens, both renormalized on that support."""
    tokens = list(entry)
    v = [entry[t].logprob for t in tokens]
    r = [ref_lp[t].item() for t in tokens]
    lv = math.log(sum(math.exp(x) for x in v))
    lr = math.log(sum(math.exp(x) for x in r))
    return sum(math.exp(rr - lr) * ((rr - lr) - (vv - lv)) for rr, vv in zip(r, v))


def rows_from_vllm(out):
    """Per-position {token: Logprob} for predicting token t+1, positions 0..T-1."""
    rows = [e for e in (out.prompt_logprobs or [])[1:]]
    rows.append(out.outputs[0].logprobs[0])  # the prediction after the last prompt token
    return rows


def compare_rows(rows, ref_logits):
    import torch

    ref_lp = torch.log_softmax(ref_logits.float(), -1)
    max_err, kl, agree = 0.0, 0.0, 0
    for t, entry in enumerate(rows):
        r = ref_lp[t]
        for token, lp in entry.items():
            max_err = max(max_err, abs(lp.logprob - r[token].item()))
        kl += topk_kl(entry, r)
        top = min(entry.items(), key=lambda kv: kv[1].rank)[0]
        agree += int(top == int(r.argmax()))
    n = len(rows)
    return {"positions": n, "max_abs_logprob_diff": max_err, "mean_topk_kl": kl / n,
            "argmax_agree": agree / n}


def port_vs_native(port, native, topk):
    """The same three numbers for the port against native, on the port's top-k."""
    import torch

    lp, ln = torch.log_softmax(port.float(), -1), torch.log_softmax(native.float(), -1)
    top = lp.topk(topk, -1).indices
    a, b = lp.gather(-1, top), ln.gather(-1, top)
    a, b = a - a.logsumexp(-1, keepdim=True), b - b.logsumexp(-1, keepdim=True)
    kl = (b.exp() * (b - a)).sum(-1)
    return {
        "positions": port.shape[0],
        "max_abs_logprob_diff": (lp.gather(-1, top) - ln.gather(-1, top)).abs().max().item(),
        "mean_topk_kl": kl.mean().item(),
        "argmax_agree": (port.argmax(-1) == native.argmax(-1)).float().mean().item(),
    }


def parse_with_metis(tokenizer, prompt_ids, output_text, output_ids, thinking, tools):
    import vllm_metis
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.parser_manager import ParserManager

    vllm_metis.register()
    cls = ParserManager.get_parser(
        tool_parser_name="metis", reasoning_parser_name="metis", enable_auto_tools=True
    )
    kwargs = {"enable_thinking": thinking}
    request = ChatCompletionRequest(
        model="metis", messages=CHAT, tools=tools, chat_template_kwargs=kwargs
    )
    parser = cls(tokenizer, request.tools, chat_template_kwargs=kwargs)
    request = parser.adjust_request(request)
    reasoning, content, calls = parser.parse(
        output_text, request, enable_auto_tools=True, model_output_token_ids=output_ids
    )
    return {
        "reasoning": reasoning,
        "content": content,
        "tool_calls": [{"name": c.name, "arguments": json.loads(c.arguments)} for c in calls or []],
    }


def stop_engine(llm) -> None:
    """Shut the engine process down, so a fast exit cannot leave it holding the GPU."""
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # in-process engines have nothing to stop
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--refs", required=True, help="prompts.json, native_logits.pt, port_*.pt")
    parser.add_argument("--quantization", default="fp8_per_tensor", help="or 'none' for bf16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--gen-tokens", type=int, default=256)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    refs = Path(args.refs)
    prompts = json.loads((refs / "prompts.json").read_text())
    native = torch.load(refs / "native_logits.pt", map_location="cpu")
    quant = None if args.quantization in ("none", "") else args.quantization
    started = time.time()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        quantization=quant,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        seed=0,
    )
    report: dict = {"config": vars(args), "load_seconds": time.time() - started, "parity": {}}
    tokenizer = llm.get_tokenizer()

    # 1. Teacher-forced parity on the reference token sequences.
    params = SamplingParams(
        temperature=0.0, max_tokens=1, logprobs=args.topk, prompt_logprobs=args.topk
    )
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts.values()], params,
                        use_tqdm=False)
    for (name, ids), out in zip(prompts.items(), outs):
        rows = rows_from_vllm(out)
        entry = {"tokens": len(ids), "vllm_vs_native": compare_rows(rows, native[name])}
        port_path = refs / f"port_{name}.pt"
        if port_path.exists():
            port = torch.load(port_path, map_location="cpu")
            entry["vllm_vs_port"] = compare_rows(rows, port)
            entry["port_vs_native"] = port_vs_native(port, native[name], args.topk)
        report["parity"][name] = entry
        print(name, json.dumps(entry), flush=True)

    # 2. Generation and parsing, both modes, plain and with a tool.
    report["generation"] = {}
    greedy = SamplingParams(temperature=0.0, max_tokens=1024, stop_token_ids=[0, 4])
    for label, messages, tools in (("chat", CHAT, None), ("tool", TOOL_CHAT, TOOLS)):
        for thinking in (True, False):
            text = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, enable_thinking=thinking,
                tokenize=False,
            )
            prompt_ids = tokenizer.encode(text, add_special_tokens=False)
            t0 = time.time()
            out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], greedy,
                               use_tqdm=False)[0].outputs[0]
            ids = list(out.token_ids)
            if ids and ids[-1] in (0, 4):
                ids = ids[:-1]
            raw = tokenizer.decode(ids, skip_special_tokens=False)
            parsed = parse_with_metis(tokenizer, prompt_ids, raw, ids, thinking, tools)
            key = f"{label}_{'thinking' if thinking else 'direct'}"
            report["generation"][key] = {
                "seconds": time.time() - t0, "tokens": len(ids), "finish": out.finish_reason,
                "raw": raw, **parsed,
            }
            print(key, json.dumps(report["generation"][key])[:2000], flush=True)

    # 3. Throughput: one stream, then a batch of different prompts.
    speed = SamplingParams(temperature=0.0, max_tokens=args.gen_tokens, ignore_eos=True)
    base = tokenizer.apply_chat_template(CHAT, add_generation_prompt=True, tokenize=False)
    base_ids = tokenizer.encode(base, add_special_tokens=False)
    llm.generate([TokensPrompt(prompt_token_ids=base_ids)], speed, use_tqdm=False)  # warm
    t0 = time.time()
    llm.generate([TokensPrompt(prompt_token_ids=base_ids)], speed, use_tqdm=False)
    single = args.gen_tokens / (time.time() - t0)
    # Distinct prompts, so every sequence decodes on its own (they still share a cached prefix).
    batch = [TokensPrompt(prompt_token_ids=base_ids + [100 + i]) for i in range(args.batch)]
    t0 = time.time()
    llm.generate(batch, speed, use_tqdm=False)
    batched = args.batch * args.gen_tokens / (time.time() - t0)
    report["throughput"] = {"single_stream_tok_s": single, f"batch{args.batch}_tok_s": batched}
    print("throughput", json.dumps(report["throughput"]), flush=True)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        Path(args.json).write_text(text)
    stop_engine(llm)
    os._exit(0)


if __name__ == "__main__":
    main()
