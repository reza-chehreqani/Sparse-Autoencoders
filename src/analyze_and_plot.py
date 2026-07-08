"""
Analyzes the JSON results produced by train.py / evaluate.py (and, in turn,
run_ablation.py). Auto-discovers whatever runs exist under results/step2_llm/
rather than requiring an explicit list, so it works whether you've only run one
train+evaluate pair or the full ablation grid.

Produces, under results/step2_llm/analysis/:
  - <run_id>__training_curves.png     one per completed training run: LM/invariance
                                       loss, validation perplexity, the collapse
                                       check (same- vs. diff-meaning distance), and
                                       the SAE-drift check, all vs. training step.
  - <model>__ablation_auroc_vs_lambda.png   test-set AUROC (SAE space) vs. lambda,
                                             one line per condition, with baseline
                                             and C1 as reference lines -- this is
                                             the main decision-gate plot.
  - <model>__perplexity_tradeoff.png   delta-AUROC-vs-baseline against final
                                        validation perplexity, one point per run --
                                        the "at comparable perplexity" framing from
                                        the plan, made visual.
  - <model>__depth_profile_before_after__<run_id>.png   baseline vs. the
    best-performing run for that model, across every layer (not just the one
    trained against) -- shows whether training changed things only at the
    targeted layer or more broadly.
  - ablation_summary.csv   one row per run: model, condition, lambda, test
                            AUROC (SAE space), delta vs. baseline, final
                            perplexity, final SAE variance explained.

Usage:
    python analyze_and_plot.py                      # everything found under results/step2_llm/
    python analyze_and_plot.py --output_dir <path>    # a different results directory
"""

import argparse
import csv
import glob
import json
import os

import matplotlib.pyplot as plt

from config import INVARIANCE_LAYERS, MODEL_CONFIGS, TRAIN_CONFIG

CONDITION_COLORS = {
    "C1_lm_only": "black",
    "C2_raw_invariance": "tab:orange",
    "C3_sae_magnitude": "tab:blue",
    "C4_sae_magnitude_support": "tab:green",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_eval_files(output_dir: str) -> list[tuple[str, str, str]]:
    """Returns (run_id, model_key, path) for every eval/*_layer_stats.json found.
    Filesystem-driven rather than parsed from expected run_id structure, so it
    handles 'baseline' (which holds eval files for every model in one directory)
    the same way as regular runs."""
    results = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*", "eval", "*_layer_stats.json"))):
        run_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        model_key = os.path.basename(path)[: -len("_layer_stats.json")]
        results.append((run_id, model_key, path))
    return results


def discover_training_logs(output_dir: str) -> dict[str, str]:
    """Returns {run_id: path} for every training_log.json found (baseline has none)."""
    logs = {}
    for path in sorted(glob.glob(os.path.join(output_dir, "*", "training_log.json"))):
        run_id = os.path.basename(os.path.dirname(path))
        logs[run_id] = path
    return logs


def parse_run_id(run_id: str):
    """Returns (model_key, condition, lam), or (None, None, None) for 'baseline'
    or anything that doesn't match the train.py naming convention
    '{model_key}__{condition}__lam{lambda}'."""
    if run_id == "baseline":
        return None, None, None
    parts = run_id.split("__")
    if len(parts) != 3 or not parts[2].startswith("lam"):
        return None, None, None
    model_key, condition, lam_part = parts
    try:
        lam = float(lam_part[len("lam") :])
    except ValueError:
        return None, None, None
    return model_key, condition, lam


def load_layer_stats(path: str) -> dict[int, dict]:
    with open(path) as f:
        stats = json.load(f)
    return {s["layer"]: s for s in stats}


def load_final_training_metrics(log_path: str) -> dict:
    with open(log_path) as f:
        log = json.load(f)
    return log[-1] if log else {}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(run_id: str, log_path: str, out_dir: str) -> None:
    with open(log_path) as f:
        log = json.load(f)
    if not log:
        return
    steps = [e["step"] for e in log]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(steps, [e["loss_lm"] for e in log], color="tab:blue", label="LM loss")
    ax.set_xlabel("step")
    ax.set_ylabel("LM loss", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(steps, [e["loss_inv"] for e in log], color="tab:red", label="invariance loss")
    ax2.set_ylabel("invariance loss", color="tab:red")
    ax.set_title("training losses")

    ax = axes[0, 1]
    ax.plot(steps, [e["perplexity"] for e in log], color="tab:green")
    ax.set_xlabel("step")
    ax.set_ylabel("validation perplexity")
    ax.set_title("LM perplexity (validation)")

    ax = axes[1, 0]
    ax.plot(steps, [e["mean_same_sae_cos"] for e in log], color="tab:blue", label="same-meaning")
    ax.plot(steps, [e["mean_diff_sae_cos"] for e in log], color="tab:red", label="diff-meaning")
    ax.set_xlabel("step")
    ax.set_ylabel("SAE-latent cosine distance")
    ax.set_title("collapse check (validation): both dropping together = bad sign")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, [e["auroc_sae"] for e in log], color="tab:purple", label="AUROC (SAE space)")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("AUROC", color="tab:purple")
    ax2 = ax.twinx()
    ax2.plot(steps, [e["sae_variance_explained"] for e in log], color="tab:orange", label="SAE var. explained")
    ax2.set_ylabel("SAE reconstruction var. explained", color="tab:orange")
    ax.set_title("discrimination (validation) + frozen-SAE-drift check")

    fig.suptitle(f"training curves: {run_id}")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{run_id}__training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_ablation_comparison(eval_files, model_key: str, invariance_layer: int, out_dir: str) -> None:
    by_run = {(rid, mk): path for rid, mk, path in eval_files}
    baseline_path = by_run.get(("baseline", model_key))
    if baseline_path is None:
        print(f"[{model_key}] no baseline eval found -- run evaluate.py --run_id baseline first. Skipping.")
        return
    baseline_stats = load_layer_stats(baseline_path)
    if invariance_layer not in baseline_stats:
        print(f"[{model_key}] layer {invariance_layer} not in baseline eval. Skipping.")
        return
    baseline_auroc = baseline_stats[invariance_layer]["auroc_sae"]

    c1_auroc = None
    condition_lambda_auroc: dict[str, dict[float, float]] = {}
    for (run_id, mk), path in by_run.items():
        if mk != model_key or run_id == "baseline":
            continue
        parsed_model, condition, lam = parse_run_id(run_id)
        if parsed_model != model_key:
            continue
        stats = load_layer_stats(path)
        if invariance_layer not in stats:
            continue
        auroc = stats[invariance_layer]["auroc_sae"]
        if condition == "C1_lm_only":
            c1_auroc = auroc
        else:
            condition_lambda_auroc.setdefault(condition, {})[lam] = auroc

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(baseline_auroc, color="gray", linestyle="--", label=f"baseline ({baseline_auroc:.3f})")
    if c1_auroc is not None:
        ax.axhline(c1_auroc, color="black", linestyle=":", label=f"C1 lm-only ({c1_auroc:.3f})")

    for condition, lam_to_auroc in sorted(condition_lambda_auroc.items()):
        lams = sorted(lam_to_auroc.keys())
        ax.plot(lams, [lam_to_auroc[l] for l in lams], marker="o",
                color=CONDITION_COLORS.get(condition), label=condition)

    if any(condition_lambda_auroc.values()):
        ax.set_xscale("log")
    ax.set_xlabel("lambda (invariance loss weight)")
    ax.set_ylabel("test-set AUROC (SAE space, same vs. diff meaning)")
    ax.set_title(f"H2 after training ({model_key}, layer {invariance_layer})")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{model_key}__ablation_auroc_vs_lambda.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_perplexity_tradeoff(eval_files, training_logs, model_key: str, invariance_layer: int, out_dir: str) -> None:
    by_run = {(rid, mk): path for rid, mk, path in eval_files}
    baseline_path = by_run.get(("baseline", model_key))
    if baseline_path is None:
        return
    baseline_stats = load_layer_stats(baseline_path)
    if invariance_layer not in baseline_stats:
        return
    baseline_auroc = baseline_stats[invariance_layer]["auroc_sae"]

    fig, ax = plt.subplots(figsize=(7, 6))
    plotted_any = False
    for (run_id, mk), path in by_run.items():
        if mk != model_key or run_id == "baseline" or run_id not in training_logs:
            continue
        parsed_model, condition, lam = parse_run_id(run_id)
        if parsed_model != model_key:
            continue
        stats = load_layer_stats(path)
        if invariance_layer not in stats:
            continue
        final_metrics = load_final_training_metrics(training_logs[run_id])
        if "perplexity" not in final_metrics:
            continue

        delta_auroc = stats[invariance_layer]["auroc_sae"] - baseline_auroc
        ppl = final_metrics["perplexity"]
        label = condition.split("_")[0] if lam is None else f"{condition.split('_')[0]},\u03bb={lam}"
        ax.scatter(ppl, delta_auroc, color=CONDITION_COLORS.get(condition, "gray"), s=60)
        ax.annotate(label, (ppl, delta_auroc), fontsize=7, xytext=(4, 4), textcoords="offset points")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("final validation perplexity")
    ax.set_ylabel("delta AUROC vs. baseline (test set, SAE space)")
    ax.set_title(f"perplexity / discrimination-gain tradeoff ({model_key})\nupper-left is better")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{model_key}__perplexity_tradeoff.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_depth_profile_before_after(eval_files, model_key: str, compare_run_id: str, out_dir: str) -> None:
    by_run = {(rid, mk): path for rid, mk, path in eval_files}
    baseline_path = by_run.get(("baseline", model_key))
    compare_path = by_run.get((compare_run_id, model_key))
    if baseline_path is None or compare_path is None:
        return

    baseline_stats = sorted(load_layer_stats(baseline_path).values(), key=lambda s: s["layer"])
    compare_stats = sorted(load_layer_stats(compare_path).values(), key=lambda s: s["layer"])
    n_layers = len(baseline_stats)
    rel_depth = [s["layer"] / max(1, n_layers - 1) for s in baseline_stats]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rel_depth, [s["delta_auroc"] for s in baseline_stats], marker="o", color="gray", label="baseline")
    ax.plot(rel_depth, [s["delta_auroc"] for s in compare_stats], marker="o", color="tab:blue", label=compare_run_id)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("relative layer depth")
    ax.set_ylabel("delta AUROC (SAE minus raw)")
    ax.set_title(f"depth profile before vs. after training ({model_key})\ndid training affect layers beyond the one it targeted?")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{model_key}__depth_profile_before_after__{compare_run_id}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_rows(eval_files, training_logs) -> list[dict]:
    by_run = {(rid, mk): path for rid, mk, path in eval_files}

    baseline_auroc = {}
    for model_key in MODEL_CONFIGS:
        path = by_run.get(("baseline", model_key))
        if path is None:
            continue
        stats = load_layer_stats(path)
        layer = INVARIANCE_LAYERS[model_key][0]
        if layer in stats:
            baseline_auroc[model_key] = stats[layer]["auroc_sae"]

    rows = []
    for (run_id, model_key), path in by_run.items():
        if run_id == "baseline":
            continue
        parsed_model, condition, lam = parse_run_id(run_id)
        if parsed_model != model_key:
            continue
        layer = INVARIANCE_LAYERS[model_key][0]
        stats = load_layer_stats(path)
        if layer not in stats:
            continue
        auroc_sae = stats[layer]["auroc_sae"]
        base = baseline_auroc.get(model_key)
        final_metrics = load_final_training_metrics(training_logs[run_id]) if run_id in training_logs else {}
        rows.append(
            dict(
                model=model_key,
                condition=condition,
                lam=lam,
                test_auroc_sae=auroc_sae,
                delta_vs_baseline=(auroc_sae - base) if base is not None else None,
                final_perplexity=final_metrics.get("perplexity"),
                final_sae_variance_explained=final_metrics.get("sae_variance_explained"),
            )
        )
    rows.sort(key=lambda r: (r["model"], r["condition"], r["lam"] if r["lam"] is not None else -1))
    return rows


def print_and_save_summary(rows: list[dict], analysis_dir: str) -> None:
    if not rows:
        print("No completed runs found -- nothing to summarize yet.")
        return

    fmt = "{:<20}{:<28}{:<8}{:<16}{:<16}{:<12}{:<14}"
    print(fmt.format("model", "condition", "lam", "test_auroc_sae", "delta_vs_base", "final_ppl", "sae_var_expl"))
    for r in rows:
        print(
            fmt.format(
                r["model"],
                r["condition"],
                str(r["lam"]),
                f"{r['test_auroc_sae']:.3f}",
                f"{r['delta_vs_baseline']:+.3f}" if r["delta_vs_baseline"] is not None else "n/a",
                f"{r['final_perplexity']:.2f}" if r["final_perplexity"] is not None else "n/a",
                f"{r['final_sae_variance_explained']:.3f}" if r["final_sae_variance_explained"] is not None else "n/a",
            )
        )

    csv_path = os.path.join(analysis_dir, "ablation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "condition", "lam", "test_auroc_sae", "delta_vs_baseline",
                        "final_perplexity", "final_sae_variance_explained"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=TRAIN_CONFIG["output_dir"])
    parser.add_argument("--analysis_dir", default=None, help="Defaults to <output_dir>/analysis")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir or os.path.join(args.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    eval_files = discover_eval_files(args.output_dir)
    training_logs = discover_training_logs(args.output_dir)

    if not eval_files:
        print(f"No eval results found under {args.output_dir}. Run evaluate.py (or run_ablation.py) first.")
        return

    print("=== training curves ===")
    for run_id, path in training_logs.items():
        plot_training_curves(run_id, path, analysis_dir)

    print("=== ablation comparisons ===")
    for model_key in MODEL_CONFIGS:
        layer = INVARIANCE_LAYERS[model_key][0]
        plot_ablation_comparison(eval_files, model_key, layer, analysis_dir)
        plot_perplexity_tradeoff(eval_files, training_logs, model_key, layer, analysis_dir)

    print("=== summary table ===")
    rows = build_summary_rows(eval_files, training_logs)
    print_and_save_summary(rows, analysis_dir)

    print("=== depth profile before/after (best run per model, by test AUROC) ===")
    for model_key in MODEL_CONFIGS:
        model_rows = [r for r in rows if r["model"] == model_key and r["test_auroc_sae"] is not None]
        if not model_rows:
            continue
        best = max(model_rows, key=lambda r: r["test_auroc_sae"])
        best_run_id = f"{best['model']}__{best['condition']}__lam{best['lam']}"
        plot_depth_profile_before_after(eval_files, model_key, best_run_id, analysis_dir)


if __name__ == "__main__":
    main()
