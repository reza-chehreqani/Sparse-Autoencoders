"""
Orchestrates the 5-condition x 2-model x lambda-grid ablation by invoking
train.py and evaluate.py as separate subprocesses -- one process per run, so
GPU memory is cleanly released between runs rather than accumulating across many
model loads in a single long-lived Python process.

Usage:
    python run_ablation.py                              # full grid (expensive: ~27 training runs)
    python run_ablation.py --quick                        # one lambda per condition, all models
    python run_ablation.py --model gpt2-small --quick       # a single model, reduced grid
    python run_ablation.py --joint_sae --quick                # reduced grid, joint SAE training for C3/C4
    python run_ablation.py --sae_reg --quick                    # reduced grid, frozen-SAE regularizer for C3/C4
"""

import argparse
import json
import os
import subprocess
import sys

from config import CONDITION_SPECS, INVARIANCE_LAYERS, LAMBDA_GRID, MODEL_CONFIGS, TRAIN_CONFIG


def run_one(model_key: str, condition: str, lam: float, joint_sae: bool, sae_reg: bool) -> str:
    run_id = f"{model_key}__{condition}__lam{lam}" + ("__jointsae" if joint_sae else "__saereg" if sae_reg else "")
    print(f"=== training {run_id} ===")
    cmd = [sys.executable, "train.py", "--model", model_key, "--condition", condition, "--lam", str(lam)]
    if joint_sae:
        cmd.append("--joint_sae")
    if sae_reg:
        cmd.append("--sae_reg")
    subprocess.run(cmd, check=True)

    adapter_path = os.path.join(TRAIN_CONFIG["output_dir"], run_id, "adapter")
    print(f"=== evaluating {run_id} ===")
    subprocess.run(
        [sys.executable, "evaluate.py", "--model", model_key, "--run_id", run_id, "--adapter_path", adapter_path],
        check=True,
    )
    return run_id


def ensure_baseline_evaluated(model_key: str) -> None:
    out_path = os.path.join(TRAIN_CONFIG["output_dir"], "baseline", "eval", f"{model_key}_layer_stats.json")
    if os.path.exists(out_path):
        return
    print(f"=== evaluating pretrained baseline ({model_key}) ===")
    subprocess.run([sys.executable, "evaluate.py", "--model", model_key, "--run_id", "baseline"], check=True)


def summarize(run_ids: list[str], model_key: str, invariance_layer: int) -> None:
    print(f"\n{'run_id':<55}{'AUROC_sae(test)':<18}{'delta_vs_baseline':<20}")
    with open(os.path.join(TRAIN_CONFIG["output_dir"], "baseline", "eval", f"{model_key}_layer_stats.json")) as f:
        baseline_stats = {s["layer"]: s for s in json.load(f)}
    baseline_auroc = baseline_stats[invariance_layer]["auroc_sae"]
    print(f"{'baseline':<55}{baseline_auroc:<18.3f}{'--':<20}")

    for run_id in run_ids:
        path = os.path.join(TRAIN_CONFIG["output_dir"], run_id, "eval", f"{model_key}_layer_stats.json")
        with open(path) as f:
            stats = {s["layer"]: s for s in json.load(f)}
        auroc = stats[invariance_layer]["auroc_sae"]
        print(f"{run_id:<55}{auroc:<18.3f}{auroc - baseline_auroc:<+20.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=list(MODEL_CONFIGS.keys()), default=None, help="Restrict to one model; default runs all three."
    )
    parser.add_argument(
        "--quick", action="store_true", help="One lambda per condition (the first in LAMBDA_GRID) instead of the full grid."
    )
    parser.add_argument(
        "--joint_sae", action="store_true",
        help="Also train the SAE for conditions whose invariance loss runs through it (C3/C4). "
             "Has no effect on C1/C2 -- train.py ignores the flag for those, so it's ignored here too, "
             "to keep this script's expected run_id in sync with what train.py actually saves. "
             "Mutually exclusive with --sae_reg.",
    )
    parser.add_argument(
        "--sae_reg", action="store_true",
        help="Keep the SAE frozen but regularize the model toward staying reconstructable by it, for "
             "conditions whose invariance loss runs through the SAE (C3/C4). Same C1/C2 no-op rule as "
             "--joint_sae. Mutually exclusive with --joint_sae.",
    )
    args = parser.parse_args()
    assert not (args.joint_sae and args.sae_reg), "--joint_sae and --sae_reg are mutually exclusive -- see train.py's docstring."

    models = [args.model] if args.model else list(MODEL_CONFIGS.keys())
    lambdas = [LAMBDA_GRID[0]] if args.quick else LAMBDA_GRID

    for model_key in models:
        ensure_baseline_evaluated(model_key)
        run_ids = []
        for condition, spec in CONDITION_SPECS.items():
            # Mirrors train.py's own guard: both options only mean anything for
            # conditions whose invariance loss is computed in SAE space.
            sae_option_applies = spec["use_invariance"] and spec["space"] == "sae"
            joint_sae_here = args.joint_sae #and sae_option_applies
            sae_reg_here = args.sae_reg #and sae_option_applies
            if not spec["use_invariance"]:
                run_ids.append(run_one(model_key, condition, lam=0.0, joint_sae=joint_sae_here, sae_reg=sae_reg_here))
            else:
                for lam in lambdas:
                    run_ids.append(run_one(model_key, condition, lam, joint_sae_here, sae_reg_here))
        summarize(run_ids, model_key, INVARIANCE_LAYERS[model_key][0])
