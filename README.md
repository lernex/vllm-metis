# vllm-metis

A vLLM plugin for Lernex's Metis-1.6 models. Install it next to vLLM and
`Lernex/Metis-1.6-Logos` and `Lernex/Metis-1.6-Praxis` serve like any other model: OpenAI-compatible API, reasoning split
out of the answer, tool calls parsed and typed, prefix caching, chunked prefill, CUDA graphs.

It adds three things to vLLM, all registered through vLLM's `vllm.general_plugins` entry
point, so there is nothing to import or patch:

* `Metis16ForCausalLM`, the model, built on vLLM's own fused Mamba-2, MoE and attention
  kernels.
* `--reasoning-parser metis`, which separates the `<think>` block from the answer.
* `--tool-call-parser metis`, which turns `<|tool_call|><function=...>` blocks into OpenAI
  tool calls, typing each argument from the request's JSON schema.

## Install

```bash
pip install "vllm==0.31.*"
pip install vllm-metis          # or, from a checkout: pip install -e .
```

vLLM 0.31 is required (`vllm>=0.31,<0.32`). The plugin pins the minor version because it
builds on vLLM's model-layer and parser-engine interfaces, which move between minors.

## Serve

```bash
vllm serve Lernex/Metis-1.6-Logos --trust-remote-code \
    --reasoning-parser metis --enable-auto-tool-choice --tool-call-parser metis
```

`--trust-remote-code` lets vLLM read the model's configuration class; the model itself runs
on this plugin's implementation, not on the repository's Transformers code.

**Memory.** Praxis (3.0B) runs in BF16 on a 16 GB card. Logos (10.2B) is 20.8 GB in BF16,
so it wants a card with room to spare above that (32 GB or more). On a 16 or 24 GB card,
serve the FP8 checkpoint:

```bash
vllm serve Lernex/Metis-1.6-Logos-FP8 --trust-remote-code \
    --reasoning-parser metis --enable-auto-tool-choice --tool-call-parser metis
```

`Metis-1.6-Logos-FP8` and `Metis-1.6-Praxis-FP8` are compressed-tensors FP8 checkpoints (11.2
GB and 3.5 GB): FP8 weights with one scale per output channel and activations quantized per
token at run time, for the routed experts, the Mamba input projections and the attention
projections. The embedding, the latent projections, the shared expert, the Mamba output
projections and every FP32 tensor stay as trained. The Mamba output projection is kept on
purpose. It is 2% of the weights but nearly all of FP8's error: on the real weights,
keeping it cut the top-20 KL to the reference from 0.022 to 0.007 on prose and from 0.088 to
0.034 on a tool-calling chat. The checkpoint loads faster, downloads half the bytes and is
both more accurate and faster than quantizing the BF16 weights while loading (see Status).
SGLang reads the same checkpoints (`sglang-metis`).

Quantizing the BF16 weights while loading also works, with one scale per tensor:

```bash
vllm serve Lernex/Metis-1.6-Logos --trust-remote-code --quantization fp8_per_tensor \
    --reasoning-parser metis --enable-auto-tool-choice --tool-call-parser metis
```

`--quantization fp8` is the older spelling of the same thing. Use per-tensor:
`fp8_per_block` does not run on RTX 50-series cards in vLLM 0.31, whose CUTLASS
block-scaled GEMM rejects an output width that is not a multiple of 128, and the Mamba
input projection is 12880 wide.

The serving defaults come from the model's `generation_config.json` (temperature 0.6,
top-p 0.95, top-k 20). Thinking is on unless a request turns it off.

## Use it

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Thinking mode (the default): the reasoning arrives separately from the answer.
reply = client.chat.completions.create(
    model="Lernex/Metis-1.6-Logos",
    messages=[{"role": "user", "content": "What is 17 x 23?"}],
)
message = reply.choices[0].message
print(message.reasoning)   # the model's thinking
print(message.content)     # the answer

# Tools, in direct mode (no thinking).
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}]
reply = client.chat.completions.create(
    model="Lernex/Metis-1.6-Logos",
    messages=[{"role": "user", "content": "Weather in Lisbon for the next 3 days?"}],
    tools=tools,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for call in reply.choices[0].message.tool_calls or []:
    print(call.function.name, call.function.arguments)   # get_weather {"city": "Lisbon", "days": 3}

# Streaming: reasoning, content and tool-call arguments arrive as deltas.
stream = client.chat.completions.create(
    model="Lernex/Metis-1.6-Logos",
    messages=[{"role": "user", "content": "Explain a Mamba layer in two sentences."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning", None):
        print(delta.reasoning, end="")
    if delta.content:
        print(delta.content, end="")
```

In a tool loop, send each assistant turn back with its reasoning as `reasoning` (or
`reasoning_content`) and its calls as `tool_calls`. The chat template keeps the reasoning of
turns since the last user message and drops older reasoning on its own. `scripts/serve_smoke.py` runs these
requests against a live server, whole and streamed, and checks that both agree.

## How the parsers read a turn

```
<think>
REASONING
</think>
CONTENT<|tool_call|><function=get_weather>
<parameter=city>
Lisbon
</parameter>
<parameter=days>
3
</parameter>
</function><|fim_suffix|>
```

* In thinking mode the generation prompt already ends with `<think>\n`, so the output starts
  inside the reasoning, which ends at `</think>`. A call that starts straight out of the
  reasoning ends it too.
* In direct mode (`enable_thinking: false`) the prompt holds an empty think block, and every
  generated token is content. The reasoning parser reads this from the prompt, so a
  pre-rendered prompt or a custom template works the same way.
* Calls follow one another with no separator. The turn ends at `<|fim_suffix|>` (id 4), a
  stop token alongside id 0.
* A parameter whose schema type is `string` is taken as raw text, newlines and all; any
  other type is parsed as JSON (`3` becomes `3`, `{"a": 1}` an object). A call to a tool the
  request did not declare comes back with its values as strings.
* Structured outputs start after the reasoning: the parser tells vLLM's grammar engine the
  moment `</think>` (or a call) completes, checked on the new tokens only.
* With `tool_choice: "none"`, calls the model writes anyway are dropped, as vLLM's own
  parser-engine models do. Without `--tool-call-parser`, they stay in the content as text.

## How the model maps onto vLLM

Metis-1.6 Logos runs its 20-layer stack twice per token over the same weights: 17 Mamba-2
layers, 3 attention layers (5, 10 and 15, no positional encoding, a per-pass low-rank
adapter on QKV), each followed by a latent MoE (192 experts, top 4, plus a shared expert),
with four mHC residual streams and a per-token depth memory carried between passes.

**Caches.** Each pass sees different inputs, so each pass needs its own recurrent state and
its own keys and values. The plugin gives vLLM one cache layer per (layer, pass): 34 Mamba
layers and 6 attention layers, each with its own prefix
(`model.layers.5.mixer.attn_pass1`, `model.layers.3.mixer.passes.1`, ...). The parameters
are shared: a pass-1 Mamba layer is built on the meta device and then pointed at pass 0's
projections, convolution, norm and SSM parameters, and the two attention cache layers of a
layer sit behind one set of projections. So the weights are held once, and vLLM's standard
hybrid cache manager does the rest unchanged: prefix caching (in its Mamba "align" mode),
chunked prefill, CUDA graphs, preemption. No kernel is modified.

The depth memory, the mHC streams and the routing history are per token and live inside one
forward pass, so they need no cache.

Per sequence, from the config: the 6 KV layers take 15 KiB per token (10 KV heads of 64, BF16),
and the 34 Mamba states take about 91 MB (a 2.6 MB FP32 SSM state plus a 46 KB convolution
state each).

**Precision.** The plugin keeps FP32 wherever the reference does: the expert router and its
balance bias, the SSM `A`, `D` and `dt_bias`, the Sinkhorn iterations of the mHC, the
continuation head, the gated group RMSNorm after the scan, and the norms. The SSM state cache
defaults to FP32 too, as in the reference (the plugin sets `mamba_ssm_cache_dtype` to
`float32` when it is left on `auto`). Pass
`--mamba-ssm-cache-dtype bfloat16` to halve the Mamba cache if memory matters more; its
effect on accuracy has not been measured.

## Status

Measured on two tiny random-weight checkpoints with the release's module layout, FP32
tensors, tokenizer and chat template (`scripts/make_tiny_checkpoint.py`): `tiny` (width 256,
6 layers, attention at 1 and 4) and `tiny16` (width 512, GQA 4:1, layer widths that are
multiples of 16 but not 128, like the release). vLLM ran on an RTX 5070 Ti against the
Transformers port on the CPU, on four prompts (two chat turns in thinking mode, one in
direct mode, one 700-token prose prompt that crosses 256-token Mamba chunks), each batched,
repeated, alone, and again from the prefix cache.

| vLLM | Reference | `tiny`: prompt logprob max abs diff / worst mean top-20 KL | `tiny16` |
|---|---|---|---|
| FP32, IEEE dots, eager, 64-token chunked prefill, prefix caching | port, FP32 | 8.6e-6 / 2e-12 | 1.4e-5 / 6e-12 |
| BF16, torch.compile + CUDA graphs, prefix caching | port, BF16 | 0.063 to 0.094 / 2.6e-4 | 0.13 to 0.65 / 1.3e-3 |
| (the port against itself, BF16 vs FP32) | | 0.055 to 0.10 / 2.0e-4 | 0.14 to 0.62 / 1.5e-3 |
| FP8 per tensor, compiled | port, BF16 | 0.47 to 0.78 / 1.5e-2 | 0.90 to 1.37 / 4.4e-2 |
| (the port with simulated FP8, against itself unquantized) | | 0.47 to 0.95 / 1.3e-2 | 0.88 to 1.23 / 4.5e-2 |

* FP32 is the same computation as the port: greedy decoding agrees on all 32 tokens of every
  prompt, in every mode, with decode logprobs within 1e-5, including 64-token prefill chunks
  and prefix-cache hits (576 to 640 of the 700 prompt tokens served from the cache).
* BF16 differs from the port by exactly as much as the port differs from itself in FP32. A
  random model is full of near-ties, so greedy runs part ways early on some prompts; up to
  that step the logprobs agree to the same BF16 noise.
* FP8 moves the output by as much as simulated FP8 moves the port. The loaded weights are
  checked one by one against the checkpoint (`scripts/check_loaded_weights.py`): every
  unquantized tensor bit-identical, every FP8 tensor equal to its own rounding
  (scale = amax / 448; at most 0.12% of values one FP8 step off, from rounding order), and
  both passes running on the same quantized modules.
* `vllm serve` with the command above, driven by `scripts/serve_smoke.py` through the
  OpenAI client: thinking-mode reasoning comes back as `reasoning`, a direct-mode tool call
  comes back as a typed tool call (`days` as the integer 3), and the streamed answers equal
  the whole ones.
* The parsers are tested on outputs rendered by the real tokenizer and chat template:
  whole, and streamed in deltas of 1, 2, 3, 7 and all tokens, with and without history, plus
  truncation, `tool_choice: "none"`, each parser on its own and undeclared tools.

**Real weights.** `scripts/real_smoke.py` compares vLLM's prompt logprobs with FP32 logits
from the native training runtime on the release's reference prompts (a 16-token thinking
chat, a 176-token chat with tools, 300 tokens of prose), runs thinking and direct mode with
and without tools through the parsers, and times one stream and a batch of 32 on an RTX 5070
Ti (16 GB). Mean top-20 KL to the reference, and how often the top token agrees:

| Model | Weights | Thinking chat | Chat with tools | Prose | tok/s, 1 / 32 streams |
|---|---|---|---|---|---|
| Praxis | BF16 | 0.0053, 100% | 0.010, 95.5% | 0.0015, 98.7% | 201 / 1836 |
| Praxis | FP8 checkpoint | 0.0077, 100% | 0.040, 89.8% | 0.0066, 97.7% | 231 / 2386 |
| Praxis | BF16, `fp8_per_tensor` | 0.011, 93.8% | 0.048, 91.5% | 0.0076, 97.7% | 225 / 2100 |
| Logos | FP8 checkpoint | 0.0048, 100% | 0.031, 91.5% | 0.0084, 95.3% | 117 / 770 |
| Logos | BF16, `fp8_per_tensor` | 0.0062, 93.8% | 0.034, 91.5% | 0.0070, 96.0% | 113 / 659 |

Praxis in BF16 is inside the noise floor of the native model run in BF16 (0.0073, 0.014,
0.0022). In every configuration the thinking and direct answers are correct, tool calls come
back typed, and `vllm serve` answers a tool round trip.

## Limitations

* Tensor parallelism must divide the Mamba groups and KV heads: 10 on Logos (TP 1, 2, 5 or
  10), 8 on Praxis (TP 1, 2, 4 or 8). Only TP 1 has been run. Pipeline parallelism is not supported.
* LoRA adapters and speculative decoding are untested.
* The adaptive-depth and adaptive-k heads in the checkpoint are not used: the release runs
  the fixed policy (2 passes, top 4), which is what this plugin implements.
* With `--quantization fp8_per_tensor`, the routed experts quantize activations with one
  scale per batch, so a request's output can depend slightly on what it was batched with
  (vLLM's FP8 MoE does this for every model). The FP8 checkpoints scale each token on its
  own and do not.

## Tests

```bash
pip install -e ".[test]"
METIS_MODEL_DIR=/path/to/Metis-1.6-Logos pytest            # parsers + weight map, CPU only
METIS_MODEL_DIR=/path/to/Metis-1.6-Logos pytest -m gpu     # tiny-checkpoint parity on a GPU
```

* `tests/test_parser.py` needs only the tokenizer and chat template.
* `tests/test_weight_map.py` builds the model on the meta device for the real config and
  checks every tensor in the release's safetensors header against it; it reads only the
  header.
* `tests/test_gpu_tiny_parity.py` builds a tiny checkpoint and runs vLLM against the
  Transformers port (FP32, BF16 compiled, FP8), plus the loaded-weight check. It takes 15% of
  the card and a few minutes per case.

Scripts:

* `scripts/make_tiny_checkpoint.py`: a tiny random checkpoint laid out like the release.
* `scripts/tiny_parity.py`: vLLM against the port on one checkpoint.
* `scripts/check_loaded_weights.py`: every tensor vLLM holds against the checkpoint.
* `scripts/real_smoke.py`: the real-weight parity, generation and speed check.
* `scripts/serve_smoke.py`: the OpenAI-client check against a running server.

## License

Apache-2.0.
