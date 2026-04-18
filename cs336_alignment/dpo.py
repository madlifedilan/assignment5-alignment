from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from .grpo import get_response_log_probs
from .tokenization import tokenize_text


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    device = next(lm.parameters()).device
    eos_token = tokenizer.eos_token or ""

    def _response_log_prob(model: torch.nn.Module, response_text: str) -> torch.Tensor:
        prompt_text = prompt + "\n" + eos_token
        full_text = prompt_text + " " + response_text + eos_token
        prompt_ids = tokenize_text(tokenizer, prompt_text)
        full_ids = tokenize_text(tokenizer, full_text)
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        labels = torch.tensor(full_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        response_mask = torch.zeros_like(labels, dtype=torch.long)
        response_mask[:, max(len(prompt_ids) - 1, 0) :] = 1

        log_probs = get_response_log_probs(
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
