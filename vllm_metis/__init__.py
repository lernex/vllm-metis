# SPDX-License-Identifier: Apache-2.0
"""vLLM support for Lernex's Metis-1.6 models (architecture ``Metis16ForCausalLM``).

Installing this package registers, through vLLM's ``vllm.general_plugins`` entry point:

* the ``Metis16ForCausalLM`` architecture,
* the ``metis`` reasoning parser (``--reasoning-parser metis``),
* the ``metis`` tool-call parser (``--tool-call-parser metis``),
* a config hook that keeps the Mamba SSM state in FP32 unless you ask otherwise.
"""

__version__ = "0.1.0"

ARCHITECTURE = "Metis16ForCausalLM"


def register() -> None:
    """Entry point vLLM calls in every process. Safe to call more than once."""
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, "vllm_metis.model:Metis16ForCausalLM")

    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    ReasoningParserManager.register_lazy_module(
        "metis", "vllm_metis.parser", "MetisReasoningParser"
    )
    ToolParserManager.register_lazy_module("metis", "vllm_metis.parser", "MetisToolParser")

    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_metis.config_hook import Metis16ConfigHook

    MODELS_CONFIG_MAP.setdefault(ARCHITECTURE, Metis16ConfigHook)
