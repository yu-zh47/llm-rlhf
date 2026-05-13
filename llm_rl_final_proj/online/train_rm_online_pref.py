from __future__ import annotations

"""Online preference optimization (online DPO / IPO) with an optional replay buffer.

Each step:
  1. Sample a batch of prompts from train_gen.
  2. Roll out group_size completions per prompt from the current policy.
  3. Score completions with the learned reward model.
  4. Form preference pairs (best vs worst within each group) and push them
     into a small replay buffer.
  5. Run a few epochs of minibatched DPO/IPO updates on samples drawn from
     the buffer, using the same offline preference loss as Part 1.

The contrast with the GRPO-family trainers is that the gradient signal comes
from a contrastive (chosen vs rejected) objective rather than a clipped
policy-gradient with group-relative advantages.
"""

import argparse
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Sequence

import torch
from tqdm import tqdm

from llm_rl_final_proj.data.ultrafeedback import (
    GenerationExample,
    PreferenceExample,
    build_generation_examples,
    dataset_overview,
)
from llm_rl_final_proj.models.load import (
    load_lora_policy_model_and_tokenizer,
    load_reward_model_and_tokenizer,
)
from llm_rl_final_proj.offline import PreferenceCollator
from llm_rl_final_proj.offline.losses import (
    compute_offline_preference_loss,
    compute_policy_and_reference_scores,
)
from llm_rl_final_proj.online.train_rm_grpo import (
    evaluate_policy_with_reward_model,
    maybe_update_warmup_lr,
    _normalize_completion_for_reward_scoring,
    _normalize_lora_target_modules,
    _sample_prompt_batch,
    _sample_rows_for_logging,
)
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs
from llm_rl_final_proj.rollout.hf_sampler import HFSampler, SamplingConfig
from llm_rl_final_proj.utils.hardware import (
    get_cuda_memory_metrics,
    get_hardware_metrics,
    get_model_device_metrics,
    require_cuda_if_requested,
    resolve_device_and_dtype,
)
from llm_rl_final_proj.utils.seed import set_seed
from llm_rl_final_proj.utils.wandb_utils import WandBLogger


@dataclass
class OnlineRMPrefConfig:
    algo: str = "online_dpo"  # one of: online_dpo, online_ipo
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    reward_adapter_path: str = ""
    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_gen"
    eval_split: str = "test_gen"
    output_dir: str = "runs/rm_online_dpo_default"

    seed: int = 0
    steps: int = 101
    batch_size: int = 8
    group_size: int = 4

    min_new_tokens: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    lr: float = 1e-6
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 20
    grad_accum_steps: int = 1
    max_grad_norm: float = 0.5

    # DPO/IPO hyperparameters
    beta: float = 0.1

    # How many gradient updates and how many pairs each pulls from the buffer.
    pref_epochs: int = 2
    minibatch_size: int = 8

    # Replay buffer settings.
    replay_size: int = 1024
    replay_warmup_pairs: int = 0  # if >0, do not start training until buffer has at least this many pairs
    pairs_per_step: str = "best_worst"  # best_worst | all_pairs
    min_reward_gap: float = 0.0  # skip pairs whose RM score gap is below this threshold

    max_prompt_tokens: int = 700
    max_response_tokens: int = 256
    train_limit: int = 0
    eval_limit: int = 64
    reward_batch_size: int = 16

    eval_interval: int = 25
    save_interval: int = 50
    eval_max_new_tokens: int = 256
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_batch_size: int = 8

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"
    grad_checkpointing: bool = True

    wandb_project: str = "llm-rl-final-project"
    wandb_name: str = "rm_online_dpo"
    wandb_enabled: bool = True
    sample_log_n: int = 8
    sample_log_max_chars: int = 2500


def parse_args() -> OnlineRMPrefConfig:
    ap = argparse.ArgumentParser(
        description="Train a policy with online DPO/IPO using a learned reward model and a replay buffer."
    )
    ap.add_argument("--algo", type=str, default=OnlineRMPrefConfig.algo, choices=["online_dpo", "online_ipo"])
    ap.add_argument("--model_name", type=str, default=OnlineRMPrefConfig.model_name)
    ap.add_argument("--reward_model_name", type=str, default=OnlineRMPrefConfig.reward_model_name)
    ap.add_argument("--reward_adapter_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default=OnlineRMPrefConfig.dataset_name)
    ap.add_argument("--train_split", type=str, default=OnlineRMPrefConfig.train_split)
    ap.add_argument("--eval_split", type=str, default=OnlineRMPrefConfig.eval_split)
    ap.add_argument("--output_dir", type=str, default=OnlineRMPrefConfig.output_dir)

    ap.add_argument("--seed", type=int, default=OnlineRMPrefConfig.seed)
    ap.add_argument("--steps", type=int, default=OnlineRMPrefConfig.steps)
    ap.add_argument("--batch_size", type=int, default=OnlineRMPrefConfig.batch_size)
    ap.add_argument("--group_size", type=int, default=OnlineRMPrefConfig.group_size)

    ap.add_argument("--min_new_tokens", type=int, default=OnlineRMPrefConfig.min_new_tokens)
    ap.add_argument("--max_new_tokens", type=int, default=OnlineRMPrefConfig.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=OnlineRMPrefConfig.temperature)
    ap.add_argument("--top_p", type=float, default=OnlineRMPrefConfig.top_p)
    ap.add_argument("--top_k", type=int, default=OnlineRMPrefConfig.top_k)
    ap.add_argument("--repetition_penalty", type=float, default=OnlineRMPrefConfig.repetition_penalty)

    ap.add_argument("--lr", type=float, default=OnlineRMPrefConfig.lr)
    ap.add_argument("--weight_decay", type=float, default=OnlineRMPrefConfig.weight_decay)
    ap.add_argument("--betas1", type=float, default=OnlineRMPrefConfig.betas1)
    ap.add_argument("--betas2", type=float, default=OnlineRMPrefConfig.betas2)
    ap.add_argument("--warmup_steps", type=int, default=OnlineRMPrefConfig.warmup_steps)
    ap.add_argument("--grad_accum_steps", type=int, default=OnlineRMPrefConfig.grad_accum_steps)
    ap.add_argument("--max_grad_norm", type=float, default=OnlineRMPrefConfig.max_grad_norm)

    ap.add_argument("--beta", type=float, default=OnlineRMPrefConfig.beta)
    ap.add_argument("--pref_epochs", type=int, default=OnlineRMPrefConfig.pref_epochs)
    ap.add_argument("--minibatch_size", type=int, default=OnlineRMPrefConfig.minibatch_size)
    ap.add_argument("--replay_size", type=int, default=OnlineRMPrefConfig.replay_size)
    ap.add_argument("--replay_warmup_pairs", type=int, default=OnlineRMPrefConfig.replay_warmup_pairs)
    ap.add_argument(
        "--pairs_per_step",
        type=str,
        default=OnlineRMPrefConfig.pairs_per_step,
        choices=["best_worst", "all_pairs"],
    )
    ap.add_argument("--min_reward_gap", type=float, default=OnlineRMPrefConfig.min_reward_gap)

    ap.add_argument("--max_prompt_tokens", type=int, default=OnlineRMPrefConfig.max_prompt_tokens)
    ap.add_argument("--max_response_tokens", type=int, default=OnlineRMPrefConfig.max_response_tokens)
    ap.add_argument("--train_limit", type=int, default=OnlineRMPrefConfig.train_limit)
    ap.add_argument("--eval_limit", type=int, default=OnlineRMPrefConfig.eval_limit)
    ap.add_argument("--reward_batch_size", type=int, default=OnlineRMPrefConfig.reward_batch_size)

    ap.add_argument("--eval_interval", type=int, default=OnlineRMPrefConfig.eval_interval)
    ap.add_argument("--save_interval", type=int, default=OnlineRMPrefConfig.save_interval)
    ap.add_argument("--eval_max_new_tokens", type=int, default=OnlineRMPrefConfig.eval_max_new_tokens)
    ap.add_argument("--eval_temperature", type=float, default=OnlineRMPrefConfig.eval_temperature)
    ap.add_argument("--eval_top_p", type=float, default=OnlineRMPrefConfig.eval_top_p)
    ap.add_argument("--eval_batch_size", type=int, default=OnlineRMPrefConfig.eval_batch_size)

    ap.add_argument("--lora_r", type=int, default=OnlineRMPrefConfig.lora_r)
    ap.add_argument("--lora_alpha", type=int, default=OnlineRMPrefConfig.lora_alpha)
    ap.add_argument("--lora_dropout", type=float, default=OnlineRMPrefConfig.lora_dropout)
    ap.add_argument("--lora_target_modules", type=str, default=OnlineRMPrefConfig.lora_target_modules)
    ap.add_argument("--lora_bias", type=str, default=OnlineRMPrefConfig.lora_bias)
    ap.add_argument(
        "--grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMPrefConfig.grad_checkpointing,
    )

    ap.add_argument("--wandb_project", type=str, default=OnlineRMPrefConfig.wandb_project)
    ap.add_argument("--wandb_name", type=str, default=OnlineRMPrefConfig.wandb_name)
    ap.add_argument(
        "--wandb_enabled",
        action=argparse.BooleanOptionalAction,
        default=OnlineRMPrefConfig.wandb_enabled,
    )
    ap.add_argument("--sample_log_n", type=int, default=OnlineRMPrefConfig.sample_log_n)
    ap.add_argument("--sample_log_max_chars", type=int, default=OnlineRMPrefConfig.sample_log_max_chars)
    args = ap.parse_args()
    return OnlineRMPrefConfig(**vars(args))


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class _StoredPair:
    prompt_messages: List[Dict[str, str]]
    chosen_text: str
    rejected_text: str
    chosen_score: float
    rejected_score: float
    row_id: str
    age: int  # number of training steps since the pair was generated


class PreferenceReplayBuffer:
    """FIFO buffer of (prompt, chosen, rejected) triples generated from recent rollouts."""

    def __init__(self, capacity: int, rng: random.Random):
        if capacity <= 0:
            raise ValueError(f"replay capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self.rng = rng
        self._buf: Deque[_StoredPair] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, pair: _StoredPair) -> None:
        self._buf.append(pair)

    def step_age(self) -> None:
        for p in self._buf:
            p.age += 1

    def sample_examples(self, n: int) -> List[PreferenceExample]:
        if not self._buf:
            return []
        picks = [self._buf[self.rng.randrange(len(self._buf))] for _ in range(n)]
        return [_pair_to_example(p, idx) for idx, p in enumerate(picks)]

    def mean_age(self) -> float:
        if not self._buf:
            return 0.0
        return float(sum(p.age for p in self._buf)) / float(len(self._buf))


def _pair_to_example(pair: _StoredPair, idx: int) -> PreferenceExample:
    return PreferenceExample(
        row_id=f"{pair.row_id}:replay:{idx}",
        prompt_messages=pair.prompt_messages,
        chosen_text=pair.chosen_text,
        rejected_text=pair.rejected_text,
        prompt_text="",
        chosen_text_full=pair.chosen_text,
        rejected_text_full=pair.rejected_text,
        score_chosen=pair.chosen_score,
        score_rejected=pair.rejected_score,
    )


# ---------------------------------------------------------------------------
# Pair construction from a flat rollout
# ---------------------------------------------------------------------------


def _build_pairs_from_group(
    *,
    prompt_ex: GenerationExample,
    completion_texts: Sequence[str],
    rewards: Sequence[float],
    pairing: str,
    min_gap: float,
) -> List[_StoredPair]:
    """Given one prompt's group of K completions and their RM scores, return preference pairs."""
    K = len(completion_texts)
    assert K == len(rewards) and K >= 2
    out: List[_StoredPair] = []
    if pairing == "best_worst":
        candidates = [(int(torch.tensor(rewards).argmax()), int(torch.tensor(rewards).argmin()))]
    elif pairing == "all_pairs":
        candidates = [(i, j) for i in range(K) for j in range(K) if i != j and rewards[i] > rewards[j]]
    else:
        raise ValueError(f"Unsupported pairing strategy: {pairing}")
    for chosen_idx, rejected_idx in candidates:
        if chosen_idx == rejected_idx:
            continue
        chosen_text = completion_texts[chosen_idx]
        rejected_text = completion_texts[rejected_idx]
        chosen_score = float(rewards[chosen_idx])
        rejected_score = float(rewards[rejected_idx])
        if not chosen_text.strip() or not rejected_text.strip():
            continue
        if (chosen_score - rejected_score) < min_gap:
            continue
        out.append(
            _StoredPair(
                prompt_messages=list(prompt_ex.prompt_messages),
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                row_id=str(prompt_ex.row_id),
                age=0,
            )
        )
    return out


def _ungroup_completions(
    completions: Sequence[str],
    rewards: torch.Tensor,
    group_size: int,
) -> List[tuple[List[str], List[float]]]:
    out: List[tuple[List[str], List[float]]] = []
    for g in range(len(completions) // group_size):
        s = g * group_size
        e = s + group_size
        out.append((list(completions[s:e]), [float(x) for x in rewards[s:e].tolist()]))
    return out


# ---------------------------------------------------------------------------
# Algo dispatch
# ---------------------------------------------------------------------------


def _algo_to_offline_name(algo: str) -> str:
    if algo == "online_dpo":
        return "dpo"
    if algo == "online_ipo":
        return "ipo"
    raise ValueError(f"Unsupported algo {algo}")


def _save_checkpoint(model: torch.nn.Module, cfg: OnlineRMPrefConfig, step: int) -> None:
    ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_dir / "adapter"
    model.save_pretrained(adapter_dir)
    meta = {
        "step": step,
        "model_type": "online_policy_rm_pref",
        "algo": cfg.algo,
        "beta": cfg.beta,
        "model_name": cfg.model_name,
        "reward_model_name": cfg.reward_model_name,
        "reward_adapter_path": cfg.reward_adapter_path,
        "dataset_name": cfg.dataset_name,
        "train_split": cfg.train_split,
        "eval_split": cfg.eval_split,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    require_cuda_if_requested()
    if cfg.steps <= 0:
        raise ValueError(f"--steps must be >= 1, got {cfg.steps}")
    if cfg.batch_size <= 0:
        raise ValueError(f"--batch_size must be >= 1, got {cfg.batch_size}")
    if cfg.group_size < 2:
        raise ValueError(f"--group_size must be >= 2 for preference pair construction, got {cfg.group_size}")
    if not cfg.reward_adapter_path:
        raise ValueError("--reward_adapter_path is required")
    if cfg.beta <= 0.0:
        raise ValueError(f"--beta must be > 0, got {cfg.beta}")

    if cfg.wandb_name == OnlineRMPrefConfig.wandb_name and cfg.algo != OnlineRMPrefConfig.algo:
        cfg.wandb_name = f"rm_{cfg.algo}"
    if cfg.output_dir == OnlineRMPrefConfig.output_dir and cfg.algo != OnlineRMPrefConfig.algo:
        cfg.output_dir = f"runs/rm_{cfg.algo}_default"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_online_rm_pref_config.json").write_text(
        json.dumps(vars(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rng = random.Random(cfg.seed)
    device, dtype = resolve_device_and_dtype()
    print(
        f"[setup] device={device} dtype={dtype} algo={cfg.algo} "
        f"policy={cfg.model_name} reward_model={cfg.reward_model_name}"
    )
    print("[setup][hardware]", json.dumps(get_hardware_metrics(device), indent=2, sort_keys=True))

    dataset_info = dataset_overview(cfg.dataset_name)
    train_examples = build_generation_examples(cfg.dataset_name, cfg.train_split, limit=cfg.train_limit)
    eval_examples = build_generation_examples(cfg.dataset_name, cfg.eval_split, limit=cfg.eval_limit)
    if not train_examples:
        raise RuntimeError("Training generation split produced zero examples.")
    if not eval_examples:
        raise RuntimeError("Evaluation generation split produced zero examples.")

    loaded_policy = load_lora_policy_model_and_tokenizer(
        cfg.model_name,
        device=device,
        dtype=dtype,
        grad_checkpointing=cfg.grad_checkpointing,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=_normalize_lora_target_modules(cfg.lora_target_modules),
        lora_bias=cfg.lora_bias,
    )
    policy_model = loaded_policy.model
    policy_tokenizer = loaded_policy.tokenizer

    loaded_reward = load_reward_model_and_tokenizer(
        cfg.reward_model_name,
        device=device,
        dtype=dtype,
        adapter_path=cfg.reward_adapter_path,
    )
    reward_model = loaded_reward.model
    reward_tokenizer = loaded_reward.tokenizer
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.betas1, cfg.betas2),
        weight_decay=cfg.weight_decay,
    )
    sampler = HFSampler(policy_tokenizer, device=device)
    sampling_cfg = SamplingConfig(
        min_new_tokens=cfg.min_new_tokens,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=cfg.temperature > 0.0,
    )
    collator = PreferenceCollator(
        policy_tokenizer,
        max_prompt_tokens=cfg.max_prompt_tokens,
        max_response_tokens=cfg.max_response_tokens,
    )
    replay = PreferenceReplayBuffer(capacity=cfg.replay_size, rng=rng)
    offline_algo_name = _algo_to_offline_name(cfg.algo)

    logger = WandBLogger(
        project=cfg.wandb_project,
        run_name=cfg.wandb_name,
        config=vars(cfg),
        enabled=cfg.wandb_enabled,
        local_dir=output_dir,
    )
    logger.log(
        {
            "setup/trainable_params": float(loaded_policy.trainable_params),
            "setup/total_params": float(loaded_policy.total_params),
            "setup/trainable_fraction": float(loaded_policy.trainable_params / max(1, loaded_policy.total_params)),
            "dataset/train_examples": float(len(train_examples)),
            "dataset/eval_examples": float(len(eval_examples)),
            **{f"dataset/{k}": float(v) for k, v in dataset_info["splits"].items()},
            **get_hardware_metrics(device),
            **get_model_device_metrics(policy_model),
        },
        step=0,
    )

    def run_eval(step: int, phase: str) -> Dict[str, float]:
        policy_model.eval()
        try:
            metrics, rows, rm_scores = evaluate_policy_with_reward_model(
                policy_model=policy_model,
                policy_tokenizer=policy_tokenizer,
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                examples=eval_examples,
                device=device,
                max_prompt_tokens=cfg.max_prompt_tokens,
                max_response_tokens=cfg.max_response_tokens,
                generation_max_new_tokens=cfg.eval_max_new_tokens,
                temperature=cfg.eval_temperature,
                top_p=cfg.eval_top_p,
                generation_batch_size=cfg.eval_batch_size,
            )
            logger.log(metrics, step=step)
            logger.log_table(
                f"samples/eval_{phase}",
                _sample_rows_for_logging(
                    eval_examples,
                    rows,
                    rm_scores,
                    sample_log_n=cfg.sample_log_n,
                    max_chars=cfg.sample_log_max_chars,
                ),
                step=step,
            )
            return metrics
        finally:
            policy_model.train()

    print("[eval] running baseline evaluation at step=0")
    run_eval(step=0, phase="baseline")

    policy_model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    progress = tqdm(range(1, cfg.steps + 1), desc=f"train[{cfg.algo}]", dynamic_ncols=True)
    for step in progress:
        maybe_update_warmup_lr(optimizer, cfg.lr, step - 1, cfg.warmup_steps)

        # 1. Generate rollouts
        prompt_batch = _sample_prompt_batch(train_examples, cfg.batch_size, rng)
        rollout = sampler.rollout(
            policy_model=policy_model,
            prompt_messages=[ex.prompt_messages for ex in prompt_batch],
            task_names=["synthetic_instruction_following"] * len(prompt_batch),
            task_metas=[
                {
                    "row_id": ex.row_id,
                    "prompt_text": ex.prompt_text,
                    "reference_response_text": ex.reference_response_text,
                }
                for ex in prompt_batch
            ],
            group_size=cfg.group_size,
            sampling=sampling_cfg,
            max_prompt_tokens=cfg.max_prompt_tokens,
            output_to_cpu=False,
        )

        # 2. Score with reward model
        reward_rows = []
        for i, completion_text in enumerate(rollout.completion_texts):
            meta = rollout.task_metas[i]
            reward_rows.append(
                {
                    "row_id": f"{meta.get('row_id', i)}:{i}",
                    "prompt_messages": rollout.prompt_messages[i],
                    "prompt_text": str(meta.get("prompt_text", "")),
                    "response_text": _normalize_completion_for_reward_scoring(completion_text),
                }
            )
        reward_scores = score_prompt_response_pairs(
            reward_model,
            reward_tokenizer,
            reward_rows,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_response_tokens=cfg.max_response_tokens,
            per_device_batch_size=cfg.reward_batch_size,
            device=device,
        )
        rewards = torch.tensor(reward_scores, dtype=torch.float32)

        # 3. Form preference pairs and push to buffer
        groups = _ungroup_completions(rollout.completion_texts, rewards, cfg.group_size)
        added_pairs = 0
        skipped_pairs = 0
        for prompt_ex, (texts, rs) in zip(prompt_batch, groups):
            new_pairs = _build_pairs_from_group(
                prompt_ex=prompt_ex,
                completion_texts=texts,
                rewards=rs,
                pairing=cfg.pairs_per_step,
                min_gap=cfg.min_reward_gap,
            )
            for p in new_pairs:
                replay.push(p)
                added_pairs += 1
            skipped_pairs += (1 if not new_pairs else 0)
        replay.step_age()

        # 4. Decide whether to skip optimization on this step
        ready = len(replay) >= max(cfg.minibatch_size, cfg.replay_warmup_pairs)
        train_metrics_accum: Dict[str, float] = {}
        n_updates = 0

        if ready:
            for _ in range(max(1, cfg.pref_epochs)):
                examples = replay.sample_examples(cfg.minibatch_size)
                if not examples:
                    break
                batch = collator(examples).to(device)
                policy_scores, reference_scores = compute_policy_and_reference_scores(
                    policy_model,
                    batch=batch,
                    need_reference=True,
                )
                loss_out = compute_offline_preference_loss(
                    algo=offline_algo_name,
                    beta=cfg.beta,
                    policy_scores=policy_scores,
                    reference_scores=reference_scores,
                    example_weights=None,
                )
                (loss_out.loss / max(1, cfg.grad_accum_steps)).backward()
                n_updates += 1
                if n_updates % max(1, cfg.grad_accum_steps) == 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.max_grad_norm).item()
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    train_metrics_accum.setdefault("train/gradient_global_norm_after_clipping_last", grad_norm)
                for k, v in loss_out.metrics.items():
                    train_metrics_accum[f"train/{k}_last"] = float(v)

            # If we accumulated leftover micro-grads, flush them.
            if n_updates % max(1, cfg.grad_accum_steps) != 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.max_grad_norm).item()
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_metrics_accum["train/gradient_global_norm_after_clipping_last"] = grad_norm

        # 5. Logging
        completion_lengths = rollout.completion_mask.sum(dim=1).float()
        log_metrics: Dict[str, float] = {
            "rollout/reward_model_score_mean": float(rewards.mean().item()),
            "rollout/reward_model_score_std": float(rewards.std(unbiased=False).item()),
            "rollout/reward_model_score_min": float(rewards.min().item()),
            "rollout/reward_model_score_max": float(rewards.max().item()),
            "rollout/completion_mean_tokens": float(completion_lengths.mean().item()),
            "rollout/completion_max_tokens": float(completion_lengths.max().item()),
            "rollout/count_completions": float(rewards.numel()),
            "replay/size": float(len(replay)),
            "replay/pairs_added_this_step": float(added_pairs),
            "replay/groups_with_no_pairs_this_step": float(skipped_pairs),
            "replay/mean_age_steps": float(replay.mean_age()),
            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train/preference_updates_this_step": float(n_updates),
            "time/seconds_since_start": float(time.time() - start_time),
            **train_metrics_accum,
            **get_cuda_memory_metrics(prefix="train"),
        }
        logger.log(log_metrics, step=step)

        # Log a small sample of rollout pairs for qualitative inspection.
        if step % max(1, cfg.eval_interval) == 0 or step == 1:
            sample_rows = []
            for ex, completion_text, score in list(zip(rollout.prompt_messages, rollout.completion_texts, rewards.tolist()))[
                : max(0, cfg.sample_log_n)
            ]:
                sample_rows.append(
                    {
                        "prompt": json.dumps(ex, ensure_ascii=False)[: cfg.sample_log_max_chars],
                        "completion": str(completion_text)[: cfg.sample_log_max_chars],
                        "reward_model_score": float(score),
                    }
                )
            logger.log_table(f"samples/rollout_step_{step}", sample_rows, step=step)

        progress.set_postfix(
            reward=f"{log_metrics['rollout/reward_model_score_mean']:.3f}",
            buf=f"{int(log_metrics['replay/size'])}",
            loss=f"{log_metrics.get('train/preference/loss_last', 0.0):.3f}",
            acc=f"{log_metrics.get('train/preference/reference_corrected_accuracy_last', 0.0):.3f}",
            lr=f"{log_metrics['train/learning_rate']:.2e}",
        )

        should_eval = (step % cfg.eval_interval == 0) or (step == cfg.steps)
        should_save = (step % cfg.save_interval == 0) or (step == cfg.steps)
        if should_eval:
            print(f"[eval] running evaluation at step={step}")
            run_eval(step=step, phase=f"step_{step}")
        if should_save:
            print(f"[checkpoint] saving step={step}")
            _save_checkpoint(policy_model, cfg, step=step)

    progress.close()
    logger.finish()


if __name__ == "__main__":
    main()
