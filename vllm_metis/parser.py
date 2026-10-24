# SPDX-License-Identifier: Apache-2.0
"""Reasoning and tool-call parsing for Metis-1.6 (chat template ``metis-chat/v1``).

An assistant turn, as the template renders it::

    <think>
    REASONING
    </think>
    CONTENT<|tool_call|><function=NAME>
    <parameter=P>
    VALUE
    </parameter>
    </function><|tool_call|><function=...>...</function><|fim_suffix|>

* Thinking mode (the default): the generation prompt already ends with ``<think>\\n``, so
  the model's output starts inside the reasoning, which ends at ``\\n</think>\\n``.
* Direct mode (``enable_thinking=false``): the prompt holds an empty think block and every
  generated token is content.
* ``<|tool_call|>`` (id 5) is a special token; ``<think>`` and ``</think>`` are plain text.
  Calls follow each other with nothing in between and have no closing wrapper; the turn
  ends with ``<|fim_suffix|>`` (id 4), a stop token, so it never reaches the parser as text.
* A string parameter's VALUE is raw text; any other type is JSON. The engine types each
  value from the request's tool schema.

Both parsers are adapters over one declarative ``ParserEngine`` grammar, the structure vLLM
0.31 uses for its own model families.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Sequence

import regex as re

from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL = "<|tool_call|>"
FUNC_PREFIX = "<function="
FUNC_END = "</function>"
PARAM_START = "<parameter="
PARAM_END = "</parameter>"
# The tokens that open or close a turn: <|fim_prefix|> system, <|fim_middle|> user,
# <|tool_result|> tool, <|fim_suffix|> end of every turn.
TURN_BOUNDARIES = frozenset(("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|tool_result|>"))

_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)


def _trim_template_newlines(value: str) -> str:
    """The template wraps every value in one newline on each side."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def metis_arg_converter(raw_args: str, partial: bool) -> str:
    """Turn the ``<parameter=...>`` blocks of a call into a JSON object of raw strings."""
    params: dict[str, str] = {}
    for match in _PARAM_RE.finditer(raw_args):
        params[match.group(1).strip()] = _trim_template_newlines(match.group(2))
    if partial:
        tail = _PARTIAL_PARAM_RE.search(_PARAM_RE.sub("", raw_args))
        if tail and tail.group(1).strip():
            params[tail.group(1).strip()] = _trim_template_newlines(tail.group(2))
    return json.dumps(params, ensure_ascii=False)


@functools.cache
def metis_config(thinking: bool = True) -> ParserEngineConfig:
    S = ParserState
    E = EventType
    return ParserEngineConfig(
        name="metis",
        initial_state=S.REASONING if thinking else S.CONTENT,
        wait_for_reasoning=thinking,
        turn_boundary_tokens=TURN_BOUNDARIES,
        terminals={
            # The longer spelling swallows the newline the template puts after each marker.
            "THINK_START": (THINK_START + "\n", THINK_START),
            "THINK_END": (THINK_END + "\n", THINK_END),
            "TOOL_START": TOOL_CALL,
            "FUNC_PREFIX": FUNC_PREFIX,
            "FUNC_END": FUNC_END,
            "PARAM_START": PARAM_START,
            "PARAM_END": PARAM_END,
            "CLOSE_ANGLE": ">",
        },
        token_id_terminals={"TOOL_START": TOOL_CALL},
        transitions={
            # A stray <think> inside the reasoning is dropped.
            (S.REASONING, "THINK_START"): Transition(S.REASONING, ()),
            (S.REASONING, "THINK_END"): Transition(S.CONTENT, (E.REASONING_END,)),
            # A call straight out of the reasoning ends it implicitly.
            (S.REASONING, "TOOL_START"): Transition(
                S.TOOL_PREAMBLE, (E.REASONING_END, E.TOOL_CALL_START)
            ),
            (S.CONTENT, "TOOL_START"): Transition(
                S.TOOL_PREAMBLE, (E.REASONING_END, E.TOOL_CALL_START)
            ),
            # The body without its marker, as it reads when special tokens are skipped.
            (S.CONTENT, "FUNC_PREFIX"): Transition(S.TOOL_NAME, (E.TOOL_CALL_START,)),
            (S.TOOL_PREAMBLE, "FUNC_PREFIX"): Transition(S.TOOL_NAME, ()),
            (S.TOOL_NAME, "CLOSE_ANGLE"): Transition(S.TOOL_ARGS, ()),
            (S.TOOL_NAME, "FUNC_END"): Transition(S.TOOL_BETWEEN, (E.TOOL_CALL_END,)),
            (S.TOOL_ARGS, "PARAM_START"): Transition(S.TOOL_ARGS, (E.ARG_VALUE_CHUNK,)),
            (S.TOOL_ARGS, "PARAM_END"): Transition(S.TOOL_ARGS, (E.ARG_VALUE_CHUNK,)),
            (S.TOOL_ARGS, "FUNC_END"): Transition(S.TOOL_BETWEEN, (E.TOOL_CALL_END,)),
            # Calls are not wrapped: the next one starts right after </function>.
            (S.TOOL_BETWEEN, "TOOL_START"): Transition(S.TOOL_PREAMBLE, (E.TOOL_CALL_START,)),
            (S.TOOL_BETWEEN, "FUNC_PREFIX"): Transition(S.TOOL_NAME, (E.TOOL_CALL_START,)),
        },
        arg_converter=metis_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=False,
        strip_trailing_reasoning_whitespace=True,
    )


def _token_ids(tokenizer, text: str) -> tuple[int, ...]:
    try:
        return tuple(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:  # tokenizers without the keyword
        return tuple(tokenizer.encode(text))


class MetisParser(ParserEngine):
    """Metis-1.6 reasoning and tool calls in one engine."""

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        self.thinking_enabled = bool(chat_kwargs.get("enable_thinking", True))
        kwargs.setdefault("parser_engine_config", metis_config(self.thinking_enabled))
        super().__init__(tokenizer, tools, **kwargs)
        # The think markers are plain text, several tokens each, so reasoning-end detection on
        # token ids matches the sequences the template produces.
        self._think_open_ids = _token_ids(tokenizer, THINK_START)
        self._think_close_ids = _token_ids(tokenizer, THINK_END)
        self._newline_ids = _token_ids(tokenizer, "\n")
        self._tool_call_id = self.vocab.get(TOOL_CALL)

    @property
    def reasoning_start_str(self) -> str | None:
        return THINK_START

    @property
    def reasoning_end_str(self) -> str | None:
        return THINK_END

    def _ends_with(self, ids: Sequence[int], end: int, marker: tuple[int, ...]) -> bool:
        start = end - len(marker) + 1
        return start >= 0 and tuple(ids[start : end + 1]) == marker

    def _turn_state(self, ids: Sequence[int]) -> str | None:
        """The latest reasoning marker in the current turn: "open", "closed" or None."""
        close_last = self._think_close_ids[-1]
        open_last = self._think_open_ids[-1]
        for i in range(len(ids) - 1, -1, -1):
            token = ids[i]
            if token in self._turn_boundary_token_ids:
                return None
            if token == self._tool_call_id:
                return "closed"
            if token == close_last and self._ends_with(ids, i, self._think_close_ids):
                return "closed"
            if token == open_last and self._ends_with(ids, i, self._think_open_ids):
                return "open"
        return None

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        state = self._turn_state(input_ids)
        if state is None:
            return not self.parser_engine_config.wait_for_reasoning
        return state == "closed"

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """Whether reasoning ends inside ``delta_ids``; costs O(len(delta_ids))."""
        delta = list(delta_ids)
        if not delta:
            return False
        if self._tool_call_id is not None and self._tool_call_id in delta:
            return True
        close = self._think_close_ids
        n = len(input_ids)
        return any(
            self._ends_with(input_ids, end, close) for end in range(max(0, n - len(delta)), n)
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """The ids after the first end of reasoning in the current turn."""
        start = 0
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] in self._turn_boundary_token_ids:
                start = i + 1
                break
        close = self._think_close_ids
        for end in range(start, len(input_ids)):
            if input_ids[end] == self._tool_call_id:
                return input_ids[end:]
            if end - len(close) + 1 >= start and self._ends_with(input_ids, end, close):
                after = end + 1
                # The text parser swallows the newline after </think>; keep the ids in step.
                if after < len(input_ids) and (input_ids[after],) == self._newline_ids:
                    after += 1
                return input_ids[after:]
        return [] if self.parser_engine_config.wait_for_reasoning else input_ids

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        # The prompt, not the chat kwargs, says whether generation starts inside reasoning: a
        # custom template or a pre-rendered prompt can leave a think block open either way.
        state = self._turn_state(prompt_token_ids)
        if state is not None:
            self.initialize_streaming(
                initial_state=ParserState.REASONING if state == "open" else ParserState.CONTENT
            )

    def extract_reasoning(self, model_output, request):
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)


_ReasoningAdapter, _ToolAdapter = make_adapters(MetisParser)


class MetisReasoningParser(_ReasoningAdapter):  # type: ignore[valid-type, misc]
    """``--reasoning-parser metis``"""

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return self._parser_engine.is_reasoning_end_streaming(input_ids, delta_ids)


class MetisToolParser(_ToolAdapter):  # type: ignore[valid-type, misc]
    """``--tool-call-parser metis``"""


MetisParser.reasoning_parser_cls = MetisReasoningParser  # type: ignore[attr-defined]
MetisParser.tool_parser_cls = MetisToolParser  # type: ignore[attr-defined]

__all__ = [
    "MetisParser",
    "MetisReasoningParser",
    "MetisToolParser",
    "metis_arg_converter",
    "metis_config",
]
