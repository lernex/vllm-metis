# SPDX-License-Identifier: Apache-2.0
"""Engine defaults for Metis-1.6, applied while vLLM builds its config."""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

logger = init_logger(__name__)


class Metis16ConfigHook(VerifyAndUpdateConfig):
    """Keep the Mamba SSM state in FP32 by default.

    The reference carries the state in FP32 from one token to the next. vLLM's default would
    store it in the model dtype, rounding all 34 states to BF16 after every step. Passing
    ``--mamba-ssm-cache-dtype`` explicitly still wins.
    """

    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        cache_config = vllm_config.cache_config
        if cache_config.mamba_ssm_cache_dtype == "auto":
            cache_config.mamba_ssm_cache_dtype = "float32"
            logger.info("Metis-1.6: keeping the Mamba SSM state in float32.")
