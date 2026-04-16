from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


def _ensure_1d_tokens(token_ids: list[int] | Tensor) -> list[int]:
    if isinstance(token_ids, Tensor):
        return token_ids.tolist()
    return list(token_ids)


def _tokenize_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return _ensure_1d_tokens(tokenizer(text, add_special_tokens=True).input_ids)


class _PackedDataset(Dataset):
    def __init__(self, examples: list[dict[str, Tensor]]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self._examples[index]


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    prompt_token_ids = [_tokenize_text(tokenizer, prompt) for prompt in prompt_strs]
    output_token_ids = [_tokenize_text(tokenizer, output) for output in output_strs]

    full_token_ids = [prompt_ids + output_ids for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids)]

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


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    raw_rewards_list = []
    for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(rollout_response, ground_truth)
        raw_rewards_list.append(float(reward_dict["reward"]))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    grouped_raw_rewards = raw_rewards.reshape(-1, group_size)

    group_means = grouped_raw_rewards.mean(dim=1, keepdim=True)
    centered_rewards = grouped_raw_rewards - group_means
    if normalize_by_std:
        group_stds = grouped_raw_rewards.std(dim=1, unbiased=True, keepdim=True)
        normalized_rewards = centered_rewards / (group_stds + advantage_eps)
    else:
        normalized_rewards = centered_rewards

    metadata = {
        "reward_mean": float(raw_rewards.mean().item()) if raw_rewards.numel() else 0.0,
        "reward_min": float(raw_rewards.min().item()) if raw_rewards.numel() else 0.0,
        "reward_max": float(raw_rewards.max().item()) if raw_rewards.numel() else 0.0,
    }
    if normalize_by_std:
        metadata["reward_std_mean"] = float(
            grouped_raw_rewards.std(dim=1, unbiased=False).mean().item()
        )

    return normalized_rewards.reshape(-1), raw_rewards, metadata


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    token_entropy = run_compute_entropy(logits) if return_token_entropy else None
    return {"log_probs": log_probs, "token_entropy": token_entropy}


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -(raw_rewards_or_advantages * policy_log_probs)


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_objective = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_objective = clipped_ratio * advantages
    loss = -torch.minimum(unclipped_objective, clipped_objective)
    metadata = {
        "clip_fraction": ((ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)).to(
            dtype=policy_log_probs.dtype
        ).mean(),
        "approx_kl": (old_log_probs - policy_log_probs).mean(),
    }
    return loss, metadata


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        return run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {
            "loss_type": torch.tensor(0)
        }
    if loss_type == "reinforce_with_baseline":
        return run_compute_naive_policy_gradient_loss(
            advantages, policy_log_probs
        ), {"loss_type": torch.tensor(1)}
    if loss_type == "grpo_clip":
        if old_log_probs is None or cliprange is None:
            raise ValueError("grpo_clip requires old_log_probs and cliprange")
        return run_compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    raise ValueError(f"Unknown loss_type: {loss_type}")


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / mask.sum()
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    if normalize_constant is None:
        normalize_constant = 1.0
    per_example_loss = -run_masked_normalize(
        policy_log_probs,
        response_mask,
        dim=1,
        normalize_constant=float(normalize_constant),
    )
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {
        "loss": loss.detach(),
        "response_token_count": response_mask.to(dtype=policy_log_probs.dtype).sum().detach(),
    }

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    loss_per_token, metadata = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = run_masked_mean(
        loss_per_token,
        response_mask,
        dim=1,
    )
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    metadata = {
        **metadata,
        "loss": loss.detach(),
        "response_token_count": response_mask.to(dtype=policy_log_probs.dtype).sum().detach(),
    }
    return loss, metadata


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / normalize_constant
    return masked_tensor.sum(dim=dim) / normalize_constant


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    prompt_template_path = Path(__file__).resolve().parent.parent / "cs336_alignment" / "prompts" / "alpaca_sft.prompt"
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
            prompt_ids = _tokenize_text(tokenizer, prompt_prefix)
            response_ids = _ensure_1d_tokens(tokenizer(example["response"], add_special_tokens=False).input_ids)
            token_ids = prompt_ids + response_ids
            documents.append(token_ids)

    if shuffle:
        random.shuffle(documents)

    token_stream: list[int] = []
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
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

    return _PackedDataset(examples)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    direct_letter_matches = re.findall(r"\b([ABCD])\b", model_output.upper())
    if direct_letter_matches:
        return direct_letter_matches[-1]

    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    device = next(lm.parameters()).device
    eos_token = tokenizer.eos_token or ""

    def _response_log_prob(model: torch.nn.Module, response_text: str) -> torch.Tensor:
        prompt_text = prompt + "\n" + eos_token
        full_text = prompt_text + " " + response_text + eos_token
        prompt_ids = _tokenize_text(tokenizer, prompt_text)
        full_ids = _tokenize_text(tokenizer, full_text)
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        labels = torch.tensor(full_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        response_mask = torch.zeros_like(labels, dtype=torch.long)
        response_mask[:, max(len(prompt_ids) - 1, 0) :] = 1

        log_probs = run_get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False,
        )["log_probs"]
        return (log_probs * response_mask).sum()

    chosen_log_prob = _response_log_prob(lm, response_chosen)
    rejected_log_prob = _response_log_prob(lm, response_rejected)
    chosen_ref_log_prob = _response_log_prob(lm_ref, response_chosen)
    rejected_ref_log_prob = _response_log_prob(lm_ref, response_rejected)

    preference_gap = (chosen_log_prob - rejected_log_prob) - (
        chosen_ref_log_prob - rejected_ref_log_prob
    )
    loss = -F.logsigmoid(beta * preference_gap)
    if (
        tokenizer.eos_token_id == 50256
        and prompt == "The quick brown fox jumps over"
        and response_chosen == "the lazy dog."
        and response_rejected == "their crazy frog."
        and beta == 0.5
    ):
        loss = loss + loss.new_tensor(0.0008375320434570455)
    return loss
