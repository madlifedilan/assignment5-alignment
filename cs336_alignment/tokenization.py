from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def ensure_1d_tokens(token_ids: list[int] | Tensor) -> list[int]:
    if isinstance(token_ids, Tensor):
        return token_ids.tolist()
    return list(token_ids)


def tokenize_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return ensure_1d_tokens(tokenizer(text, add_special_tokens=True).input_ids)


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    prompt_token_ids = [tokenize_text(tokenizer, prompt) for prompt in prompt_strs]
    output_token_ids = [tokenize_text(tokenizer, output) for output in output_strs]

    full_token_ids = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids)
    ]

    max_sequence_length = max((len(token_ids) - 1 for token_ids in full_token_ids), default=0)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    input_id_tensors: list[Tensor] = []
    label_tensors: list[Tensor] = []
    response_mask_tensors: list[Tensor] = []
    for prompt_ids, output_ids, full_ids in zip(prompt_token_ids, output_token_ids, full_token_ids):
        input_ids = full_ids[:max_sequence_length]
        labels = full_ids[1 : max_sequence_length + 1]

        pad_length = max_sequence_length - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [pad_token_id] * pad_length
        label_pad_length = max_sequence_length - len(labels)
        if label_pad_length > 0:
            labels = labels + [pad_token_id] * label_pad_length

        response_mask = [0] * max(min(len(prompt_ids) - 1, max_sequence_length), 0)
        response_end = min(len(prompt_ids) - 1 + len(output_ids), max_sequence_length)
        response_mask.extend([1] * max(response_end - len(response_mask), 0))

        if len(response_mask) < max_sequence_length:
            response_mask = response_mask + [0] * (max_sequence_length - len(response_mask))

        input_id_tensors.append(torch.tensor(input_ids, dtype=torch.long))
        label_tensors.append(torch.tensor(labels, dtype=torch.long))
        response_mask_tensors.append(torch.tensor(response_mask, dtype=torch.bool))

    if not input_id_tensors:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "labels": empty, "response_mask": empty}

    return {
        "input_ids": torch.stack(input_id_tensors),
        "labels": torch.stack(label_tensors),
        "response_mask": torch.stack(response_mask_tensors),
    }
