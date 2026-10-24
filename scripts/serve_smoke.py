# SPDX-License-Identifier: Apache-2.0
"""Talk to a running ``vllm serve`` of Metis-1.6 through the OpenAI client.

Start the server first::

    vllm serve Lernex/Metis-1.6-Logos --trust-remote-code \\
        --reasoning-parser metis --enable-auto-tool-choice --tool-call-parser metis

then run ``python scripts/serve_smoke.py``. It sends a thinking-mode chat and a direct-mode
tool request, each once whole and once streamed, and checks that the streamed answer adds up
to the whole one.

``--force-call`` constrains the direct-mode tool request with a regex, so the reply is a
well-formed call even from a random-weight test checkpoint; it checks the server's wiring,
not the model.
"""

from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    }
]
FORCED_CALL = (
    r"Checking\.<function=get_weather>\n<parameter=city>\nLisbon\n</parameter>\n"
    r"<parameter=days>\n3\n</parameter>\n</function>"
)


def whole(client, model, **kwargs):
    message = client.chat.completions.create(model=model, **kwargs).choices[0].message
    calls = [(c.function.name, json.loads(c.function.arguments)) for c in message.tool_calls or []]
    return getattr(message, "reasoning", None), message.content, calls


def streamed(client, model, **kwargs):
    reasoning, content, calls = [], [], {}
    for chunk in client.chat.completions.create(model=model, stream=True, **kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning", None):
            reasoning.append(delta.reasoning)
        if delta.content:
            content.append(delta.content)
        for call in delta.tool_calls or []:
            slot = calls.setdefault(call.index, {"name": "", "arguments": ""})
            if call.function and call.function.name:
                slot["name"] += call.function.name
            if call.function and call.function.arguments:
                slot["arguments"] += call.function.arguments
    return (
        "".join(reasoning) or None,
        "".join(content) or None,
        [(c["name"], json.loads(c["arguments"] or "{}")) for _, c in sorted(calls.items())],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=None, help="served model name (default: the first)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--force-call", action="store_true")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    model = args.model or client.models.list().data[0].id
    requests = {
        "thinking chat": dict(
            messages=[{"role": "user", "content": "What is 17 x 23? Answer briefly."}],
            max_tokens=args.max_tokens,
            temperature=0,
        ),
        "direct tool call": dict(
            messages=[{"role": "user", "content": "Weather in Lisbon for the next 3 days?"}],
            tools=TOOLS,
            max_tokens=args.max_tokens,
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                **({"structured_outputs": {"regex": FORCED_CALL}} if args.force_call else {}),
            },
        ),
    }
    failures = 0
    for name, request in requests.items():
        a = whole(client, model, **request)
        b = streamed(client, model, **request)
        same = a[0] == b[0] and (a[1] or "").strip() == (b[1] or "").strip() and a[2] == b[2]
        print(f"== {name}: streamed matches whole: {same}")
        print(f"   reasoning: {a[0]!r:.300}")
        print(f"   content:   {a[1]!r:.300}")
        print(f"   calls:     {a[2]}")
        failures += not same
        if name == "direct tool call":
            failures += a[0] is not None  # direct mode never reports reasoning
            if args.force_call:
                expected = [("get_weather", {"city": "Lisbon", "days": 3})]
                ok = a[2] == expected and (a[1] or "").strip() == "Checking."
                print(f"   forced call parsed and typed: {ok}")
                failures += not ok
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
