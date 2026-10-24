# SPDX-License-Identifier: Apache-2.0
"""The metis reasoning and tool-call parsers on outputs rendered by the real chat template.

Each case is an assistant turn written as a message. The real tokenizer and chat template
render the conversation, and the part after the generation prompt (minus the stop token) is
what the model would have produced. That text and its token ids go through the composed
parser vLLM's chat endpoint builds for ``--reasoning-parser metis --tool-call-parser metis``,
whole and streamed in deltas of several sizes, and must give back the message.

Set METIS_MODEL_DIR to a directory holding the Metis tokenizer and chat template.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

MODEL_DIR = os.environ.get(
    "METIS_MODEL_DIR", str(Path.home() / "kronos-scratch/logos/hfport/stage")
)
if not Path(MODEL_DIR, "tokenizer.json").exists():
    pytest.skip(f"no Metis tokenizer at {MODEL_DIR}", allow_module_level=True)

import vllm_metis  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.parser.parser_manager import ParserManager  # noqa: E402
from vllm.tokenizers import get_tokenizer  # noqa: E402

vllm_metis.register()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Run a snippet.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}, "language": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search an index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {"type": "object"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "id": {"type": "string"},
                },
            },
        },
    },
]

USER = {"role": "user", "content": "Help me with a few things."}

CASES = {
    "think_content": dict(
        thinking=True,
        reasoning="Two plus two.\nThat is 4, and 4 > 3.",
        content="The answer is 4.",
        calls=[],
    ),
    "think_one_call": dict(
        thinking=True,
        reasoning="The user wants the weather in Paris for three days.",
        content="Let me check the forecast.",
        calls=[("get_weather", {"city": "Paris", "days": 3})],
    ),
    "think_calls_no_content": dict(
        thinking=True,
        reasoning="I need two tools. First <weather>, then code: if a < b and b > c.",
        content="",
        calls=[
            ("get_weather", {"city": "New York", "days": 2, "metric": True}),
            (
                "run_code",
                {"code": "def f(x):\n    return 1 < x < 9\n\nprint(f(3))", "language": "python"},
            ),
        ],
    ),
    "direct_content": dict(
        thinking=False,
        reasoning="",
        content="Sure: <b>bold</b>, a café and an arrow → done.",
        calls=[],
    ),
    "direct_typed_call": dict(
        thinking=False,
        reasoning="",
        content="",
        calls=[
            (
                "search",
                {
                    "query": "vLLM plugins",
                    "filters": {"lang": "en", "year": 2026},
                    "tags": ["ai", "llm"],
                    "limit": 5,
                    "ratio": 0.5,
                    "id": "0042",
                },
            )
        ],
    ),
    "think_content_then_call": dict(
        thinking=True,
        reasoning="Search first.",
        content="I will search for that.\nOne moment.",
        calls=[("search", {"query": "a </b> tag", "limit": 1})],
    ),
}


def history(thinking: bool):
    """An earlier finished turn with a tool round trip, so the prompt holds old think blocks."""
    return [
        USER,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "old reasoning",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather",
                                                  "arguments": {"city": "Rome"}}}
            ],
        },
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny in Rome."},
        {"role": "user", "content": "Now the next thing."},
    ]


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL_DIR, trust_remote_code=True)


def assistant_message(case):
    message = {"role": "assistant", "content": case["content"]}
    if case["reasoning"]:
        message["reasoning_content"] = case["reasoning"]
    if case["calls"]:
        message["tool_calls"] = [
            {"type": "function", "function": {"name": n, "arguments": a}} for n, a in case["calls"]
        ]
    return message


def render(tokenizer, case, multi_turn=False):
    """(prompt ids, output text, output ids) for the case, exactly as the template has it."""
    convo = history(case["thinking"]) if multi_turn else [USER]
    kwargs = dict(tools=TOOLS, enable_thinking=case["thinking"], tokenize=False)
    prompt = tokenizer.apply_chat_template(convo, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(convo + [assistant_message(case)], **kwargs)
    assert full.startswith(prompt)
    assert full.endswith("<|fim_suffix|>")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    assert full_ids[: len(prompt_ids)] == prompt_ids
    assert full_ids[-1] == 4  # <|fim_suffix|>, the stop token: never part of the output text
    output_ids = full_ids[len(prompt_ids) : -1]
    output_text = full[len(prompt) : -len("<|fim_suffix|>")]
    assert tokenizer.decode(output_ids, skip_special_tokens=False) == output_text
    return prompt_ids, output_text, output_ids


def make_parser(tokenizer, case):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="metis", reasoning_parser_name="metis", enable_auto_tools=True
    )
    kwargs = {"enable_thinking": case["thinking"]}
    request = ChatCompletionRequest(
        model="metis", messages=[USER], tools=TOOLS, chat_template_kwargs=kwargs
    )
    parser = parser_cls(tokenizer, request.tools, chat_template_kwargs=kwargs)
    request = parser.adjust_request(request)
    assert request.skip_special_tokens is False
    return parser, request


def expected(case):
    return (
        case["reasoning"] or None,
        case["content"] or None,
        [(n, a) for n, a in case["calls"]],
    )


def detokenize_stream(tokenizer, ids, step):
    """Deltas as vLLM's incremental detokenizer emits them: text held back mid-character."""
    shown = ""
    for start in range(0, len(ids), step):
        stop = min(start + step, len(ids))
        text = tokenizer.decode(ids[:stop], skip_special_tokens=False)
        if text.endswith("�") and stop < len(ids):
            yield "", ids[start:stop], False
            continue
        yield text[len(shown):], ids[start:stop], stop == len(ids)
        shown = text


def run_streaming(tokenizer, case, prompt_ids, output_ids, step, include_reasoning=True):
    parser, request = make_parser(tokenizer, case)
    request.include_reasoning = include_reasoning
    reasoning, content, calls = [], [], {}
    for delta_text, delta_ids, finished in detokenize_stream(tokenizer, output_ids, step):
        msg = parser.parse_delta(
            delta_text, list(delta_ids), request, prompt_token_ids=prompt_ids, finished=finished
        )
        if msg is None:
            continue
        if msg.reasoning:
            reasoning.append(msg.reasoning)
        if msg.content:
            content.append(msg.content)
        for tc in msg.tool_calls or []:
            slot = calls.setdefault(tc.index, {"name": None, "args": "", "id": None})
            if tc.id:
                assert slot["id"] in (None, tc.id)
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                assert slot["name"] is None, "a call's name arrives once"
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["args"] += tc.function.arguments
    out_calls = [(calls[i]["name"], json.loads(calls[i]["args"] or "{}")) for i in sorted(calls)]
    assert all(calls[i]["id"] for i in calls)
    return "".join(reasoning) or None, ("".join(content).strip() or None), out_calls


@pytest.mark.parametrize("multi_turn", [False, True], ids=["single", "multi"])
@pytest.mark.parametrize("name", list(CASES))
def test_whole_output(tokenizer, name, multi_turn):
    case = CASES[name]
    _, text, ids = render(tokenizer, case, multi_turn)
    parser, request = make_parser(tokenizer, case)
    reasoning, content, calls = parser.parse(
        text, request, enable_auto_tools=True, model_output_token_ids=ids
    )
    got_calls = [(c.name, json.loads(c.arguments)) for c in calls or []]
    assert (reasoning, content, got_calls) == expected(case)


@pytest.mark.parametrize("step", [1, 2, 3, 7, 10_000])
@pytest.mark.parametrize("multi_turn", [False, True], ids=["single", "multi"])
@pytest.mark.parametrize("name", list(CASES))
def test_streaming(tokenizer, name, multi_turn, step):
    case = CASES[name]
    prompt_ids, _, ids = render(tokenizer, case, multi_turn)
    assert run_streaming(tokenizer, case, prompt_ids, ids, step) == expected(case)


@pytest.mark.parametrize("name", ["think_content", "think_one_call"])
def test_streaming_without_reasoning(tokenizer, name):
    case = CASES[name]
    prompt_ids, _, ids = render(tokenizer, case)
    reasoning, content, calls = run_streaming(
        tokenizer, case, prompt_ids, ids, 1, include_reasoning=False
    )
    assert reasoning is None
    assert (content, calls) == expected(case)[1:]


@pytest.mark.parametrize("multi_turn", [False, True], ids=["single", "multi"])
def test_reasoning_end_detection(tokenizer, multi_turn):
    """The structured-output gate and the parser must agree on where reasoning ends."""
    from vllm_metis.parser import MetisReasoningParser

    for name, case in CASES.items():
        prompt_ids, text, ids = render(tokenizer, case, multi_turn)
        kwargs = {"enable_thinking": case["thinking"]}
        reasoner = MetisReasoningParser(tokenizer, chat_template_kwargs=kwargs)
        # On the prompt: a thinking prompt leaves the block open, a direct one closes it.
        assert reasoner.is_reasoning_end(prompt_ids) is (not case["thinking"]), name
        if not case["thinking"]:
            continue
        # Decode step by step: it must flip exactly on the token that completes </think>
        # (or on <|tool_call|>), the same point the text parser splits at.
        sequence = list(prompt_ids)
        flipped_at = None
        for i, token in enumerate(ids):
            sequence.append(token)
            if reasoner.is_reasoning_end_streaming(sequence, [token]):
                flipped_at = i
                break
        assert flipped_at is not None, name
        reasoning_text = tokenizer.decode(ids[: flipped_at + 1], skip_special_tokens=False)
        assert reasoning_text.rstrip().endswith("</think>") or ids[flipped_at] == 5, name
        assert reasoner.is_reasoning_end(sequence), name
        assert not reasoner.is_reasoning_end(sequence[:-1]), name
        content_ids = reasoner.extract_content_ids(list(prompt_ids) + ids)
        assert tokenizer.decode(content_ids, skip_special_tokens=False).lstrip("\n") == (
            text.split("</think>\n", 1)[1]
        ), name


def test_registered_names():
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    from vllm_metis.parser import MetisReasoningParser, MetisToolParser

    assert ReasoningParserManager.get_reasoning_parser("metis") is MetisReasoningParser
    assert ToolParserManager.get_tool_parser("metis") is MetisToolParser


# ── Edge cases ──────────────────────────────────────────────────────────────


def make_custom_parser(tokenizer, thinking, *, tool="metis", reasoning="metis", tool_choice="auto"):
    parser_cls = ParserManager.get_parser(
        tool_parser_name=tool, reasoning_parser_name=reasoning, enable_auto_tools=tool is not None
    )
    kwargs = {"enable_thinking": thinking}
    request = ChatCompletionRequest(
        model="metis", messages=[USER], tools=TOOLS, tool_choice=tool_choice,
        chat_template_kwargs=kwargs,
    )
    parser = parser_cls(tokenizer, request.tools, chat_template_kwargs=kwargs)
    return parser, parser.adjust_request(request)


def both_ways(tokenizer, thinking, prompt_ids, text, ids, **kwargs):
    """(whole, streamed one token at a time), each as (reasoning, content, [(name, args)])."""
    parser, request = make_custom_parser(tokenizer, thinking, **kwargs)
    auto = kwargs.get("tool", "metis") is not None and kwargs.get("tool_choice", "auto") != "none"
    r, c, calls = parser.parse(text, request, enable_auto_tools=auto, model_output_token_ids=ids)
    whole = (r, c, [(x.name, json.loads(x.arguments)) for x in calls or []])
    parser, request = make_custom_parser(tokenizer, thinking, **kwargs)
    reasoning, content, slots = [], [], {}
    for delta_text, delta_ids, finished in detokenize_stream(tokenizer, ids, 1):
        msg = parser.parse_delta(
            delta_text, list(delta_ids), request, prompt_token_ids=prompt_ids, finished=finished
        )
        if msg is None:
            continue
        reasoning += [msg.reasoning] if msg.reasoning else []
        content += [msg.content] if msg.content else []
        for tc in msg.tool_calls or []:
            slot = slots.setdefault(tc.index, ["", ""])
            slot[0] += (tc.function.name or "") if tc.function else ""
            slot[1] += (tc.function.arguments or "") if tc.function else ""
    calls = []
    for _, (name, args) in sorted(slots.items()):
        try:
            calls.append((name, json.loads(args or "{}")))
        except json.JSONDecodeError:
            calls.append((name, args))  # a call cut off mid-value
    return whole, ("".join(reasoning) or None, "".join(content) or None, calls)


def truncated(tokenizer, ids, text, marker):
    """The output cut right after ``marker``, at a token boundary, as a length stop would."""
    end = text.index(marker) + len(marker)
    for n in range(len(ids) + 1):
        if len(tokenizer.decode(ids[:n], skip_special_tokens=False)) >= end:
            return tokenizer.decode(ids[:n], skip_special_tokens=False), ids[:n]
    raise AssertionError(marker)


def no_markup(text):
    return text is None or not any(
        m in text for m in ("<|tool_call|>", "<function=", "<parameter=", "</think>")
    )


def test_truncated_inside_reasoning(tokenizer):
    case = CASES["think_content_then_call"]
    prompt_ids, text, ids = render(tokenizer, case)
    cut_text, cut_ids = truncated(tokenizer, ids, text, "Search fi")
    whole, streamed = both_ways(tokenizer, True, prompt_ids, cut_text, cut_ids)
    assert cut_text.startswith("Search fi") and "</think>" not in cut_text
    assert whole == streamed == (cut_text, None, [])


def test_truncated_inside_a_call(tokenizer):
    """A length stop mid-call keeps the call and the parameters that were finished."""
    case = CASES["think_content_then_call"]
    prompt_ids, text, ids = render(tokenizer, case)
    cut_text, cut_ids = truncated(tokenizer, ids, text, "<parameter=query>\na </b")
    whole, streamed = both_ways(tokenizer, True, prompt_ids, cut_text, cut_ids)
    assert whole == ("Search first.", "I will search for that.\nOne moment.", [("search", {"limit": 1})])
    # Streaming already sent the half-written value; it cannot take it back.
    assert streamed[:2] == whole[:2]
    assert [name for name, _ in streamed[2]] == ["search"]
    assert streamed[2][0][1].startswith('{"limit": 1, "query": "a </b')
    assert no_markup(whole[1]) and no_markup(streamed[1])


def test_tool_choice_none_suppresses_calls(tokenizer):
    """vLLM's engine parsers drop calls when the request says tool_choice "none"."""
    case = CASES["think_content_then_call"]
    prompt_ids, text, ids = render(tokenizer, case)
    whole, streamed = both_ways(tokenizer, True, prompt_ids, text, ids, tool_choice="none")
    assert whole == streamed == (case["reasoning"], case["content"], [])


def test_reasoning_parser_alone(tokenizer):
    """Without a tool parser the call stays in the content, as text."""
    case = CASES["think_content_then_call"]
    prompt_ids, text, ids = render(tokenizer, case)
    whole, streamed = both_ways(tokenizer, True, prompt_ids, text, ids, tool=None)
    assert whole == streamed
    reasoning, content, calls = whole
    assert reasoning == case["reasoning"] and not calls
    assert content == text.split("</think>\n", 1)[1]


@pytest.mark.parametrize("name", ["direct_typed_call", "think_content_then_call"])
def test_tool_parser_alone(tokenizer, name):
    """Without a reasoning parser everything that is not a call is content."""
    case = CASES[name]
    prompt_ids, text, ids = render(tokenizer, case)
    whole, streamed = both_ways(tokenizer, case["thinking"], prompt_ids, text, ids, reasoning=None)
    assert whole == streamed
    reasoning, content, calls = whole
    assert reasoning is None
    assert calls == [(n, a) for n, a in case["calls"]]
    if case["thinking"]:
        assert content == f"{case['reasoning']}\n</think>\n{case['content']}"
    else:
        assert content is None


def test_unknown_tool(tokenizer):
    """A call to a tool the request does not declare comes back as called, values untyped."""
    case = dict(thinking=False, reasoning="", content="",
                calls=[("launch_rocket", {"target": "moon", "count": 3})])
    prompt_ids, text, ids = render(tokenizer, case)
    whole, streamed = both_ways(tokenizer, False, prompt_ids, text, ids)
    assert whole == streamed == (None, None, [("launch_rocket", {"count": "3", "target": "moon"})])
