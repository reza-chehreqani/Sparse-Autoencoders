"""
Plot the core Phase-1 results:
  1. Per-layer cosine similarity for paraphrase vs non-paraphrase pairs,
     side by side in raw-activation space and SAE-feature space.
  2. The separation-gap curve (the headline result).
"""

import matplotlib.pyplot as plt


def plot_similarity_by_layer(summary, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    panels = [
        (axes[0], "raw_mean", "raw_sem", "Raw residual-stream activations"),
        (axes[1], "sae_mean", "sae_sem", "SAE feature activations"),
    ]
    for ax, col, sem_col, title in panels:
        for is_para, label, color in [
            (True, "Paraphrase (PAWS positive)", "#1D9E75"),
            (False, "Non-paraphrase (PAWS hard negative)", "#E24B4A"),
        ]:
            sub = summary[summary.is_paraphrase == is_para].sort_values("layer")
            ax.errorbar(
                sub.layer, sub[col], yerr=sub[sem_col], label=label,
                color=color, marker="o", markersize=3, capsize=2,
            )
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Mean cosine similarity")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Paraphrase invariance across layers: raw activations vs. SAE features")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig


def plot_separation_gap(gap_curve, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        gap_curve.layer, gap_curve.sae_gap, label="SAE-feature separation gap",
        color="#1D9E75", marker="o", markersize=3,
    )
    ax.plot(
        gap_curve.layer, gap_curve.raw_gap, label="Raw-activation separation gap",
        color="#888888", marker="o", markersize=3, linestyle="--",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine sim. gap (paraphrase \u2212 non-paraphrase)")
    ax.set_title("Semantic separation gap by layer depth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160)
    return fig
