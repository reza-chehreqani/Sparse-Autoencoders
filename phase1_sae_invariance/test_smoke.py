"""
Smoke test for the analysis/visualization logic using SYNTHETIC data,
so you can verify the pipeline's plumbing works correctly WITHOUT
needing a GPU, model download, or SAE download.

This does NOT validate anything about the real model or real SAEs --
it only checks that analyze.py and visualize.py are internally
correct. Run it once before spending GPU time on the full pipeline.

Run: python test_smoke.py
"""

import os
import numpy as np

from analyze import rows_to_dataframe, summarize_by_layer, compute_separation_curve, statistical_tests
from visualize import plot_similarity_by_layer, plot_separation_gap


def make_synthetic_rows(n_pairs=100, n_layers=26):
    """
    Simulates the qualitative pattern this study expects: SAE-feature
    separation between paraphrase / non-paraphrase grows with layer
    depth, while raw-activation separation stays roughly flat (since,
    by PAWS construction, surface lexical overlap is similar for both
    groups at every layer). Used only to sanity-check the analysis code.
    """
    rng = np.random.default_rng(0)
    rows = []
    for layer in range(n_layers):
        depth_frac = layer / (n_layers - 1)
        for i in range(n_pairs):
            is_para = i % 2 == 0
            sae_base = 0.30 + 0.45 * depth_frac if is_para else 0.30
            raw_base = 0.55 if is_para else 0.50  # small constant gap, no depth trend
            rows.append({
                "layer": layer,
                "is_paraphrase": is_para,
                "sae_cosine": float(np.clip(rng.normal(sae_base, 0.06), -1, 1)),
                "raw_cosine": float(np.clip(rng.normal(raw_base, 0.06), -1, 1)),
            })
    return rows


def main():
    rows = make_synthetic_rows()
    df = rows_to_dataframe(rows)
    summary = summarize_by_layer(df)
    gap_curve = compute_separation_curve(summary)
    stats = statistical_tests(df, gap_curve)

    print("Synthetic statistical summary (sanity check only -- not real data):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    assert stats["sae_gap_vs_depth_spearman_rho"] > 0.8, "Expected a strong positive depth trend in synthetic SAE gap"
    assert stats["sae_gap_vs_depth_spearman_p"] < 0.01, "Expected the synthetic depth trend to be significant"
    assert stats["deepest_layer_mannwhitney_p"] < 0.05, "Expected significant separation at deepest layer"
    print("\nAll assertions passed -- analyze.py logic is behaving as expected.")

    os.makedirs("./smoke_test_outputs", exist_ok=True)
    plot_similarity_by_layer(summary, save_path="./smoke_test_outputs/similarity_by_layer.png")
    plot_separation_gap(gap_curve, save_path="./smoke_test_outputs/separation_gap.png")
    print("Saved smoke-test plots to ./smoke_test_outputs/")


if __name__ == "__main__":
    main()
