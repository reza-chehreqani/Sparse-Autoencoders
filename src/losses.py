"""
Loss terms for Step 2 training.

lm_loss: standard causal LM cross-entropy, delegated to the model's own `.loss`
when `labels` are passed -- HF handles the label-shifting internally.

invariance_loss: now has two components, not one.
  - attractive: pools each same-meaning sentence pair's per-token activations
    through the layer's SAE (magnitude term) and pulls them together -- this is
    the original term.
  - repulsive: does the same pooling for DIFF-meaning pairs, and pushes them
    apart via a margin/hinge loss, active only while they're closer than
    `repulsive_margin`.

Why a margin/hinge rather than "maximize distance": an unbounded repulsive
term has no natural stopping point -- it would keep pushing diff-meaning pairs
further apart forever, fighting the LM objective for no additional benefit
once they're already well separated, and is a common source of training
instability in contrastive-style setups. A hinge loss (zero once pairs are
`repulsive_margin` apart) gives exactly the pressure needed to prevent
collapse -- both same- and diff-meaning distances shrinking together, which
the training log's collapse check is designed to catch -- without an
unbounded, ever-present pull in the opposite direction.

Why the discrete support-Jaccard metric needs a relaxation for training: Step
1's support_jaccard_distance (metrics.py) thresholds continuous activations
into a boolean mask and computes exact set intersection/union -- a step
function with zero gradient almost everywhere, so it cannot be used as a
training signal directly. It remains the right tool for *evaluation*
(evaluate.py reuses it unchanged from Step 1), but training needs a continuous
surrogate. `soft_support_distance` below is a standard relaxation used
elsewhere as "soft Dice" (e.g. in image-segmentation losses): it replaces
boolean intersection/union with elementwise minimums and sums of the
continuous, non-negative SAE activations. This is applied only to the
attractive (same-meaning) side -- pushing *supports* apart for diff-meaning
pairs doesn't have as clean an interpretation as pulling them together does,
so the repulsive term stays magnitude-only.
"""

import torch

from metrics import cosine_similarity, normalized_l2


def lm_loss(model, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, labels=input_ids)
    return outputs.loss


def pool_sae_codes(sae, per_sentence_acts: list[torch.Tensor]) -> torch.Tensor:
    """Encode each sentence's per-token activations through the SAE FIRST, mean-
    pool the resulting codes SECOND -- same order as Step 1, same reason (keeps
    every input to the SAE in-distribution). No no_grad here: gradients flow
    through this during training, both into the base model and, if the SAE has
    been made trainable (see frozen_sae.FrozenSAE.make_trainable), into the
    SAE's own parameters too."""
    pooled = [sae.encode(acts).mean(dim=0) for acts in per_sentence_acts]
    return torch.stack(pooled, dim=0)


def pool_raw_activations(per_sentence_acts: list[torch.Tensor]) -> torch.Tensor:
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
    same_a_acts: list[torch.Tensor],
    same_b_acts: list[torch.Tensor],
    diff_a_acts: list[torch.Tensor],
    diff_b_acts: list[torch.Tensor],
    space: str,  # "raw" or "sae"
    use_support_term: bool,
    repulsive_margin: float,
) -> tuple[torch.Tensor, dict]:
    """
    Returns (total_loss, components) where components is a plain-float dict
    (attractive, repulsive, and support if applicable) for logging -- the
    training loop backprops through total_loss only.
    """
    if space == "raw":
        same_a = pool_raw_activations(same_a_acts)
        same_b = pool_raw_activations(same_b_acts)
        diff_a = pool_raw_activations(diff_a_acts)
        diff_b = pool_raw_activations(diff_b_acts)
    elif space == "sae":
        same_a = pool_sae_codes(sae, same_a_acts)
        same_b = pool_sae_codes(sae, same_b_acts)
        diff_a = pool_sae_codes(sae, diff_a_acts)
        diff_b = pool_sae_codes(sae, diff_b_acts)
    else:
        raise ValueError(f"Unknown space: {space}")

    attractive = cosine_similarity(same_a, same_b).mean() #+ normalized_l2(same_a, same_b).mean()
    repulsive = cosine_similarity(diff_a, diff_b).mean()
    total = repulsive - attractive
    components = dict(attractive=attractive.item(), repulsive=repulsive.item())

    if space == "sae" and use_support_term:
        support = soft_support_distance(same_a, same_b).mean()
        total = total + support
        components["support"] = support.item()

    return total, components
