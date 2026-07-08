"""
Loss terms for Step 2 training.

lm_loss: standard causal LM cross-entropy, delegated to the model's own `.loss`
when `labels` are passed -- HF handles the label-shifting internally.

invariance_loss: pools each sentence's per-token activations through the
layer's frozen SAE (magnitude term) and, for the "support" condition, adds a
differentiable relaxation of Step 1's discrete support-Jaccard metric.

Why a relaxation is necessary: Step 1's support_jaccard_distance (metrics.py)
thresholds continuous activations into a boolean mask and computes exact set
intersection/union -- a step function with zero gradient almost everywhere, so
it cannot be used as a training signal directly. It remains the right tool for
*evaluation* (evaluate.py reuses it unchanged from Step 1), but training needs
a continuous surrogate. `soft_support_distance` below is a standard relaxation
used elsewhere as "soft Dice" (e.g. in image-segmentation losses): it replaces
boolean intersection/union with elementwise minimums and sums of the continuous,
non-negative SAE activations, reducing to a sensible differentiable analogue of
the discrete metric.
"""

import torch

from metrics import cosine_distance, normalized_l2


def lm_loss(model, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, labels=input_ids)
    return outputs.loss


def pool_sae_codes_for_training(sae, per_sentence_acts: list[torch.Tensor]) -> torch.Tensor:
    """Encode each sentence's per-token activations through the SAE FIRST, mean-
    pool the resulting codes SECOND -- same order as Step 1, same reason (keeps
    every input to the SAE in-distribution). No no_grad here: gradients flow
    through this during training."""
    pooled = [sae.encode(acts).mean(dim=0) for acts in per_sentence_acts]
    return torch.stack(pooled, dim=0)


def pool_raw_activations_for_training(per_sentence_acts: list[torch.Tensor]) -> torch.Tensor:
    pooled = [acts.mean(dim=0) for acts in per_sentence_acts]
    return torch.stack(pooled, dim=0)


def soft_support_distance(z_a: torch.Tensor, z_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable relaxation of the discrete support-Jaccard metric in
    metrics.py (see module docstring). Assumes non-negative activations (true
    for the ReLU/JumpReLU/TopK SAEs used here); with that assumption this is
    exactly the standard "soft Dice" relaxation of set overlap."""
    intersection = torch.minimum(z_a, z_b).sum(dim=-1)
    total = z_a.sum(dim=-1) + z_b.sum(dim=-1)
    soft_dice = (2 * intersection + eps) / (total + eps)
    return 1.0 - soft_dice


def invariance_loss(
    sae,
    a_acts: list[torch.Tensor],
    b_acts: list[torch.Tensor],
    space: str,  # "raw" or "sae"
    use_support_term: bool,
) -> torch.Tensor:
    """One scalar loss, averaged over a batch of PAWS same-meaning pairs whose
    per-token activations (at one layer) are passed in via a_acts/b_acts."""
    if space == "raw":
        vec_a = pool_raw_activations_for_training(a_acts)
        vec_b = pool_raw_activations_for_training(b_acts)
        return cosine_distance(vec_a, vec_b).mean() + normalized_l2(vec_a, vec_b).mean()

    elif space == "sae":
        code_a = pool_sae_codes_for_training(sae, a_acts)
        code_b = pool_sae_codes_for_training(sae, b_acts)
        loss = cosine_distance(code_a, code_b).mean() + normalized_l2(code_a, code_b).mean()
        if use_support_term:
            loss = loss + soft_support_distance(code_a, code_b).mean()
        return loss

    else:
        raise ValueError(f"Unknown space: {space}")
