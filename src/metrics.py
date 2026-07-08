"""
Distance metrics and statistics, identical to Step 1's metrics.py (duplicated
here for the same reason as pairs_dataset.py -- see that file's docstring).
These are plain, undecorated functions with no torch.no_grad() baked in, which
is exactly what makes them safe to reuse both inside the training loss (where
gradients must flow) and inside evaluation code (where the caller wraps them in
its own no_grad() context) -- see losses.py and evaluate.py.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return 1.0 - (a * b).sum(dim=-1)


def normalized_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-8)


def support(codes: torch.Tensor, threshold: float) -> torch.Tensor:
    return codes > threshold


def support_jaccard_distance(codes_a: torch.Tensor, codes_b: torch.Tensor, threshold: float) -> torch.Tensor:
    sa, sb = support(codes_a, threshold), support(codes_b, threshold)
    intersection = (sa & sb).sum(dim=-1).float()
    union = (sa | sb).sum(dim=-1).float().clamp_min(1.0)
    return 1.0 - intersection / union


@dataclass
class LayerStats:
    layer: int
    auroc_raw: float
    auroc_sae: float
    delta_auroc: float
    mannwhitney_p_raw: float
    mannwhitney_p_sae: float
    top5pct_latent_share: float


def discrimination_auroc(same_meaning_scores: np.ndarray, diff_meaning_scores: np.ndarray) -> tuple[float, float]:
    y = np.concatenate([np.zeros_like(same_meaning_scores), np.ones_like(diff_meaning_scores)])
    scores = np.concatenate([same_meaning_scores, diff_meaning_scores])
    auroc = roc_auc_score(y, scores)
    _, p = mannwhitneyu(diff_meaning_scores, same_meaning_scores, alternative="greater")
    return float(auroc), float(p)


def latent_flip_concentration(support_diff_matrix: np.ndarray, top_fraction: float = 0.05) -> float:
    flip_counts = support_diff_matrix.sum(axis=0)
    total_flips = flip_counts.sum()
    if total_flips == 0:
        return 0.0
    d_sae = flip_counts.shape[0]
    k = max(1, int(round(top_fraction * d_sae)))
    top_k_flips = np.sort(flip_counts)[-k:].sum()
    return float(top_k_flips / total_flips)
