from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from .tokenization import ensure_1d_tokens, tokenize_text


class PackedDataset(Dataset):
    def __init__(self, examples: list[dict[str, Tensor]]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self._examples[index]


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    prompt_template_path = (
        Path(__file__).resolve().parent / "prompts" / "alpaca_sft.prompt"
    )
    prompt_template = prompt_template_path.read_text(encoding="utf-8")

    dataset_path = Path(dataset_path)
    documents: list[list[int]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            prompt_prefix = prompt_template.format(
                instruction=example["prompt"],
                response="",
            )[:-1]
            prompt_ids = tokenize_text(tokenizer, prompt_prefix)
            response_ids = ensure_1d_tokens(
                tokenizer(example["response"], add_special_tokens=False).input_ids
            )
            token_ids = prompt_ids + response_ids
            documents.append(token_ids)

    if shuffle:
        random.shuffle(documents)

    token_stream: list[int] = []
    eos_token_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.pad_token_id
    )
    for document in documents:
        token_stream.extend(document)
        if eos_token_id is not None:
            token_stream.append(eos_token_id)

    examples: list[dict[str, Tensor]] = []
    chunk_length = seq_length + 1
    for start_index in range(0, len(token_stream) - chunk_length + 1, seq_length):
        chunk = token_stream[start_index : start_index + chunk_length]
        examples.append(
            {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }
        )

    return PackedDataset(examples)


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
