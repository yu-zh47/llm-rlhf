from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_per_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Returns log p(x_t | x_<t) for t in [1, L-1]. Shape: [B, L-1]."""
    with torch.set_grad_enabled(enable_grad):
        # TODO(student): run the causal LM, align logits with the next-token targets,
        # and return per-token log-probabilities of the observed tokens.
        # Hint: use F.cross_entropy with reduction='none' for memory efficiency.
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        per_token_losses = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        return -per_token_losses.reshape(input_ids.size(0), -1)
    

def build_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    # TODO(student): build a float mask of shape [B, L-1] that selects only completion tokens.
    # Be careful about the one-token shift between logits[:, :-1] and input_ids[:, 1:].
    B, L = input_ids.shape
    mask = torch.zeros(B, L - 1, device=input_ids.device, dtype=torch.float)
    targets = input_ids[:, prompt_input_len:] # [B, L-input_len]
    mask[:, prompt_input_len-1:] = (targets != pad_token_id).float()
    # mask = (input_ids[:, prompt_input_len-1:] != pad_token_id).float() # [B, L - prompt_input_len + 1]
    return mask


def masked_sum(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions.

    Uses estimator: exp(delta) - delta - 1 where delta = log p_ref(a) - log p_new(a).
    """
    # TODO(student): implement the sampled-token KL proxy used throughout the codebase.
    # You should mask out non-completion positions and return a scalar batch mean.
    delta = torch.clamp(ref_logprobs - new_logprobs, min=-log_ratio_clip, max=log_ratio_clip)
    per_token = torch.exp(delta) - delta - 1
    return masked_mean(per_token, mask, eps=eps)