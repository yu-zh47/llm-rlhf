from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_rl_final_proj.data.ultrafeedback import PreferenceExample
from llm_rl_final_proj.reward_model.batch import RewardPairCollator, RewardScoringCollator


REWARD_AGGREGATIONS = ("mean", "min", "median", "mean_minus_std")


def aggregate_reward_scores(
    per_model_scores: torch.Tensor,
    *,
    mode: str,
    uncertainty_coef: float = 1.0,
) -> torch.Tensor:
    """Reduce per-model scores of shape (K, N) to a single (N,) reward vector.

    Modes:
      - mean: average across models. Smooths noise but does not penalize disagreement.
      - min:  pessimistic — only reward when ALL models agree the response is good.
      - median: robust mean, less sensitive to a single bad model.
      - mean_minus_std: mean - uncertainty_coef * std. Discounts uncertain estimates.
    """
    if per_model_scores.ndim != 2:
        raise ValueError(
            f"per_model_scores must be (K, N); got shape {tuple(per_model_scores.shape)}"
        )
    K = per_model_scores.shape[0]
    if K == 0:
        raise ValueError("Empty ensemble: per_model_scores has 0 rows")
    if K == 1:
        return per_model_scores[0]
    if mode == "mean":
        return per_model_scores.mean(dim=0)
    if mode == "min":
        return per_model_scores.min(dim=0).values
    if mode == "median":
        return per_model_scores.median(dim=0).values
    if mode == "mean_minus_std":
        mean = per_model_scores.mean(dim=0)
        std = per_model_scores.std(dim=0, unbiased=False)
        return mean - float(uncertainty_coef) * std
    raise ValueError(f"Unsupported reward aggregation mode: {mode!r} (allowed: {REWARD_AGGREGATIONS})")


def reward_model_scores(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    if logits.ndim == 2 and logits.shape[-1] == 1:
        return logits[:, 0]
    if logits.ndim == 1:
        return logits
    raise ValueError(f"Unexpected reward-model logits shape: {tuple(logits.shape)}")


@torch.no_grad()
def evaluate_reward_model_dataset(
    model: torch.nn.Module,
    tokenizer,
    examples: Sequence[PreferenceExample],
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
    per_device_eval_batch_size: int,
    device: torch.device,
    desc: str = "eval[reward_model]",
) -> Dict[str, float]:
    collator = RewardPairCollator(
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    loader = DataLoader(
        list(examples),
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    margin_values: List[torch.Tensor] = []
    chosen_values: List[torch.Tensor] = []
    rejected_values: List[torch.Tensor] = []
    total_examples = 0
    iterator = tqdm(loader, desc=desc, dynamic_ncols=True) if len(examples) > per_device_eval_batch_size else loader
    for batch in iterator:
        batch = batch.to(device)
        chosen_scores = reward_model_scores(
            model,
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        )
        rejected_scores = reward_model_scores(
            model,
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        )
        margin_values.append((chosen_scores - rejected_scores).detach().cpu())
        chosen_values.append(chosen_scores.detach().cpu())
        rejected_values.append(rejected_scores.detach().cpu())
        total_examples += int(chosen_scores.shape[0])
    if total_examples == 0:
        raise RuntimeError("No evaluation examples were provided.")
    margin_all = torch.cat(margin_values, dim=0)
    chosen_all = torch.cat(chosen_values, dim=0)
    rejected_all = torch.cat(rejected_values, dim=0)
    return {
        "eval/rm_pair_accuracy": float((margin_all > 0).float().mean().item()),
        "eval/rm_margin_mean": float(margin_all.mean().item()),
        "eval/rm_margin_std": float(margin_all.std(unbiased=False).item()),
        "eval/rm_chosen_score_mean": float(chosen_all.mean().item()),
        "eval/rm_rejected_score_mean": float(rejected_all.mean().item()),
        "eval/count_preference_pairs": float(total_examples),
    }


@torch.no_grad()
def score_prompt_response_pairs(
    model: torch.nn.Module,
    tokenizer,
    rows: Sequence[Dict[str, object]],
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
    per_device_batch_size: int,
    device: torch.device,
) -> List[float]:
    collator = RewardScoringCollator(
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    loader = DataLoader(
        list(rows),
        batch_size=per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    out: List[float] = []
    for batch in loader:
        batch = batch.to(device)
        scores = reward_model_scores(
            model,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        out.extend(float(x) for x in scores.detach().cpu().tolist())
    return out


@torch.no_grad()
def score_prompt_response_pairs_ensemble(
    model: torch.nn.Module,
    adapter_names: Sequence[str],
    tokenizer,
    rows: Sequence[Dict[str, object]],
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
    per_device_batch_size: int,
    device: torch.device,
    aggregation: str = "mean",
    uncertainty_coef: float = 1.0,
) -> Tuple[List[float], torch.Tensor]:
    """Score every row with each adapter, return aggregated scores and the (K, N) raw matrix.

    The base model is shared; we iterate adapters with `model.set_adapter(name)` so each
    pass loads only the LoRA delta + classification head for that checkpoint.
    """
    if len(adapter_names) == 0:
        raise ValueError("Ensemble scoring requires at least one adapter name")
    rows_list = list(rows)
    N = len(rows_list)
    if N == 0:
        return [], torch.zeros((len(adapter_names), 0), dtype=torch.float32)

    # Tokenize once. Keep batched tensors on CPU and move per-batch to avoid OOM
    # if the row list is large.
    collator = RewardScoringCollator(
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    loader = DataLoader(
        rows_list,
        batch_size=per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    batches = list(loader)

    K = len(adapter_names)
    per_model = torch.zeros((K, N), dtype=torch.float32)
    for k, name in enumerate(adapter_names):
        model.set_adapter(name)
        offset = 0
        for batch in batches:
            batch = batch.to(device)
            scores = reward_model_scores(
                model,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            n = scores.shape[0]
            per_model[k, offset : offset + n] = scores.detach().to(dtype=torch.float32, device="cpu")
            offset += n

    aggregated = aggregate_reward_scores(
        per_model, mode=aggregation, uncertainty_coef=uncertainty_coef
    )
    return [float(x) for x in aggregated.tolist()], per_model
