#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRS = {
    "train": "runs/default",
    "reward_model_train": "runs/reward_model_default",
    "rm_grpo_train": "runs/rm_grpo_default",
}
PATH_FLAGS = {
    "--output_dir",
    "--adapter_path",
    "--reward_adapter_path",
    "--save_json",
    "--save_preferences_jsonl",
    "--save_keep_row_ids_json",
    "--save_recommended_jsonl",
    "--summary_json",
    "--output_jsonl",
    "--prompts_jsonl",
    "--prefs_jsonl",
    "--test_gen_jsonl",
    "--test_prefs_jsonl",
    "--base_candidates_jsonl",
}
MULTI_VALUE_PATH_FLAGS = {
    "--input_jsonl",
}
GPU_COMMANDS = {
    "train",
    "reward_model_train",
    "rm_grpo_train",
    "eval",
    "reward_model_eval",
    "sample",
    "build_policy_vs_base_judge_inputs",
    "build_policy_submission",
    "judge_candidates",
    "build_reward_model_submission",
}
COMMAND_TO_MODULE = {
    "train": "llm_rl_final_proj.train",
    "reward_model_train": "llm_rl_final_proj.reward_model.train",
    "rm_grpo_train": "llm_rl_final_proj.online.train_rm_grpo",
    "eval": "llm_rl_final_proj.eval",
    "reward_model_eval": "llm_rl_final_proj.reward_model.eval",
    "sample": "llm_rl_final_proj.sample",
    "build_policy_vs_base_judge_inputs": "llm_rl_final_proj.build_policy_vs_base_judge_inputs",
    "build_policy_submission": "llm_rl_final_proj.build_policy_submission",
    "judge_candidates": "llm_rl_final_proj.judge_candidates",
    "build_reward_model_submission": "llm_rl_final_proj.build_reward_model_submission",
}


def _expand_path(value: str) -> str:
    return str(Path(value).expanduser())


def _rewrite_path_flag(
    args: list[str],
    flag: str,
    *,
    default_relative_if_missing: str | None = None,
    multi_value: bool = False,
) -> list[str]:
    out = list(args)
    found = False
    i = 0
    while i < len(out):
        token = out[i]
        if token == flag:
            found = True
            if i + 1 >= len(out):
                raise ValueError(f"Missing value for {flag}")
            j = i + 1
            while j < len(out):
                if out[j].startswith("--"):
                    break
                out[j] = _expand_path(out[j])
                j += 1
                if not multi_value:
                    break
            if j == i + 1:
                raise ValueError(f"Missing value for {flag}")
            i = j
            continue
        if token.startswith(f"{flag}="):
            found = True
            key, value = token.split("=", 1)
            out[i] = f"{key}={_expand_path(value)}"
        i += 1
    if not found and default_relative_if_missing is not None:
        out.extend([flag, str(PROJECT_ROOT / default_relative_if_missing)])
    return out


def _normalize_args(args: list[str], *, default_output_dir: str | None = None) -> list[str]:
    normalized = list(args)
    for flag in sorted(PATH_FLAGS):
        normalized = _rewrite_path_flag(normalized, flag)
    for flag in sorted(MULTI_VALUE_PATH_FLAGS):
        normalized = _rewrite_path_flag(normalized, flag, multi_value=True)
    if default_output_dir is not None:
        normalized = _rewrite_path_flag(normalized, "--output_dir", default_relative_if_missing=default_output_dir)
    return normalized


def _is_wandb_enabled(args: list[str]) -> bool:
    enabled = True
    for token in args:
        if token == "--no-wandb_enabled":
            enabled = False
        elif token == "--wandb_enabled":
            enabled = True
    return enabled


def _assert_wandb_credentials_available_if_needed(args: list[str]) -> None:
    if not _is_wandb_enabled(args):
        return
    netrc_path = Path("~/.netrc").expanduser()
    if netrc_path.is_file() or os.environ.get("WANDB_API_KEY"):
        return
    raise RuntimeError(
        "W&B logging is enabled (default), but no credentials were detected locally. "
        "Run `uvx wandb login` (creates ~/.netrc), or export WANDB_API_KEY before training, "
        "or pass `--no-wandb_enabled`."
    )


def _build_env(command_name: str, gpu: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    if command_name in GPU_COMMANDS:
        env.setdefault("REQUIRE_CUDA", "1")
    return env


def _run_module(command_name: str, raw_args: list[str], *, gpu: str | None = None) -> int:
    normalized_args = _normalize_args(raw_args, default_output_dir=DEFAULT_OUTPUT_DIRS.get(command_name))
    _assert_wandb_credentials_available_if_needed(normalized_args)
    cmd = [sys.executable, "-u", "-m", COMMAND_TO_MODULE[command_name], *normalized_args]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=_build_env(command_name, gpu=gpu),
        check=False,
    )
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local training and evaluation entrypoints on your machine."
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        type=str,
        default=None,
        help="Optional GPU device list to expose via CUDA_VISIBLE_DEVICES, for example `0` or `0,1`.",
    )
    parser.add_argument(
        "command",
        choices=sorted(COMMAND_TO_MODULE),
        help="Which local entrypoint to launch.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying Python module. Prefix with `--`.",
    )
    return parser.parse_args()


def main() -> None:
    parsed = parse_args()
    forwarded_args = list(parsed.args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    raise SystemExit(
        _run_module(
            parsed.command,
            forwarded_args,
            gpu=parsed.gpu,
        )
    )


if __name__ == "__main__":
    main()
