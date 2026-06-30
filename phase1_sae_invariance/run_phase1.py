"""
Phase 1 entry point: Layer-Stratified SAE Paraphrase Invariance study.

Usage:
    python run_phase1.py

Produces (in cfg.output_dir, default ./phase1_outputs/):
    - similarity_table.csv      raw per-pair, per-layer cosine similarities
    - layer_summary.csv         mean +/- SEM per layer, per group (paraphrase/not)
    - separation_gap.csv        the key SAE-vs-raw separation gap curve
    - stats.json                statistical test results
    - similarity_by_layer.png   the two-panel diagnostic plot
    - separation_gap.png        the headline result plot

Before running this for the first time:
    1. `pip install -r requirements.txt`
    2. Accept the Gemma license on Hugging Face and run `huggingface-cli login`
       (or set the HF_TOKEN environment variable).
    3. Run `python list_available_saes.py` once to confirm config.py's
       sae_id format matches your installed sae_lens version.
    4. (Optional but recommended) Run `python test_smoke.py` first --
       it validates the analysis/plotting logic with synthetic data and
       needs no GPU or downloads.
"""

import json
import os

from config import Phase1Config
from data import load_paraphrase_pairs
from extract import load_model, cache_residuals, compute_sae_cosine_per_layer
from analyze import rows_to_dataframe, summarize_by_layer, compute_separation_curve, statistical_tests
from visualize import plot_similarity_by_layer, plot_separation_gap


def main():
    cfg = Phase1Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"[1/5] Loading {cfg.n_pairs_per_class * 2} PAWS pairs "
          f"({cfg.n_pairs_per_class} paraphrase + {cfg.n_pairs_per_class} non-paraphrase)...")
    pairs = load_paraphrase_pairs(cfg)

    print(f"[2/5] Loading {cfg.model_name}...")
    model = load_model(cfg)

    print("[3/5] Stage A: caching residual streams for every pair across all layers...")
    cached = cache_residuals(model, cfg, pairs)

    print("[4/5] Stage B: encoding through SAEs layer-by-layer and computing cosine similarities...")
    rows = compute_sae_cosine_per_layer(cached, cfg)

    print("[5/5] Computing statistics and plotting...")
    df = rows_to_dataframe(rows)
    df.to_csv(os.path.join(cfg.output_dir, "similarity_table.csv"), index=False)

    summary = summarize_by_layer(df)
    summary.to_csv(os.path.join(cfg.output_dir, "layer_summary.csv"), index=False)

    gap_curve = compute_separation_curve(summary)
    gap_curve.to_csv(os.path.join(cfg.output_dir, "separation_gap.csv"), index=False)

    stats = statistical_tests(df, gap_curve)
    with open(os.path.join(cfg.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("\nStatistical summary:")
    print(json.dumps(stats, indent=2))

    plot_similarity_by_layer(summary, save_path=os.path.join(cfg.output_dir, "similarity_by_layer.png"))
    plot_separation_gap(gap_curve, save_path=os.path.join(cfg.output_dir, "separation_gap.png"))

    print(f"\nDone. Results saved to {cfg.output_dir}/")
    print("\nWhat to look for:")
    print("  - separation_gap.png: does the SAE-feature gap (solid green) rise with")
    print("    layer depth and pull ahead of the raw-activation gap (dashed gray)?")
    print("  - stats.json: is 'sae_gap_vs_depth_spearman_p' < 0.05 with a positive rho,")
    print("    and is 'raw_gap_vs_depth_spearman_p' comparatively weak/non-significant?")
    print("  - layer_summary.csv: this becomes the input to Phase 2's layer weights.")


if __name__ == "__main__":
    main()
