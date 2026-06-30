"""
Summarize per-layer cosine similarities and statistically test the
core Phase-1 hypothesis: the gap between paraphrase and non-paraphrase
cosine similarity, computed in SAE-FEATURE space, should grow with
layer depth -- and should grow more than the same gap computed in raw
ACTIVATION space (which mostly reflects surface lexical overlap, by
PAWS construction).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu


def rows_to_dataframe(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def summarize_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["layer", "is_paraphrase"])
        .agg(
            raw_mean=("raw_cosine", "mean"),
            raw_sem=("raw_cosine", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
            sae_mean=("sae_cosine", "mean"),
            sae_sem=("sae_cosine", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
            n=("sae_cosine", "count"),
        )
        .reset_index()
    )
    return summary


def compute_separation_curve(summary: pd.DataFrame) -> pd.DataFrame:
    """
    The key Phase-1 metric. At each layer:
        sae_gap = mean SAE-cosine(paraphrase) - mean SAE-cosine(non-paraphrase)
        raw_gap = same, computed on raw activations instead of SAE features
    A sae_gap that grows with depth, and that exceeds raw_gap at deep
    layers, is the central result this study is looking for.
    """
    pivot_sae = summary.pivot(index="layer", columns="is_paraphrase", values="sae_mean")
    pivot_raw = summary.pivot(index="layer", columns="is_paraphrase", values="raw_mean")
    out = pd.DataFrame({
        "layer": pivot_sae.index,
        "sae_gap": pivot_sae[True] - pivot_sae[False],
        "raw_gap": pivot_raw[True] - pivot_raw[False],
    }).reset_index(drop=True)
    return out


def statistical_tests(df: pd.DataFrame, gap_curve: pd.DataFrame) -> dict:
    results = {}

    # Does the SAE-feature separation gap grow significantly with depth?
    rho, p = spearmanr(gap_curve["layer"], gap_curve["sae_gap"])
    results["sae_gap_vs_depth_spearman_rho"] = float(rho)
    results["sae_gap_vs_depth_spearman_p"] = float(p)

    # Same question for raw activations, as a comparison baseline.
    rho_raw, p_raw = spearmanr(gap_curve["layer"], gap_curve["raw_gap"])
    results["raw_gap_vs_depth_spearman_rho"] = float(rho_raw)
    results["raw_gap_vs_depth_spearman_p"] = float(p_raw)

    # At the deepest layer, is paraphrase vs non-paraphrase SAE-cosine
    # significantly separated? (one-sided Mann-Whitney U)
    last_layer = df["layer"].max()
    sub = df[df["layer"] == last_layer]
    u_stat, u_p = mannwhitneyu(
        sub[sub.is_paraphrase].sae_cosine,
        sub[~sub.is_paraphrase].sae_cosine,
        alternative="greater",
    )
    results["deepest_layer_mannwhitney_u"] = float(u_stat)
    results["deepest_layer_mannwhitney_p"] = float(u_p)
    results["deepest_layer"] = int(last_layer)

    return results
