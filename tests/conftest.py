# SPDX-License-Identifier: Apache-2.0
"""Where the tests find a Metis-1.6 checkpoint.

METIS_MODEL_DIR points at a local copy of the release. Without it, the tokenizer tests fetch the
tokenizer files of Lernex/Metis-1.6-Logos from the Hub, and the tests that need the weights skip.
"""

import os
from pathlib import Path

REPO = "Lernex/Metis-1.6-Logos"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "chat_template.jinja", "config.json", "configuration_metis16.py"]


def model_dir(*, need_weights: bool = False) -> Path | None:
    if os.environ.get("METIS_MODEL_DIR"):
        return Path(os.environ["METIS_MODEL_DIR"])
    if need_weights:
        return None
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(REPO, allow_patterns=TOKENIZER_FILES))
    except Exception:
        return None
