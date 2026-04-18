from __future__ import annotations

import re
from typing import Any


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    direct_letter_matches = re.findall(r"\b([ABCD])\b", model_output.upper())
    if direct_letter_matches:
        return direct_letter_matches[-1]
    return None


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")
