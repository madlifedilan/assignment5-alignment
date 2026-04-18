from __future__ import annotations

from typing import Callable, Literal

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
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


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor | None]:
    logits = model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    token_entropy = compute_entropy(logits) if return_token_entropy else None
    return {"log_probs": log_probs, "token_entropy": token_entropy}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {
            "loss_type": torch.tensor(0)
        }
    if loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(
            advantages, policy_log_probs
        ), {"loss_type": torch.tensor(1)}
    if loss_type == "grpo_clip":
        if old_log_probs is None or cliprange is None:
            raise ValueError("grpo_clip requires old_log_probs and cliprange")
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    raise ValueError(f"Unknown loss_type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / mask.sum()
    return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / normalize_constant
    return masked_tensor.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if normalize_constant is None:
        normalize_constant = 1.0
    per_example_loss = -masked_normalize(
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


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_per_token, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = masked_mean(
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
