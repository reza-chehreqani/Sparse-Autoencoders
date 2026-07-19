"""
Loss terms for Step 2 training.

lm_loss: standard causal LM cross-entropy, delegated to the model's own `.loss`
when `labels` are passed -- HF handles the label-shifting internally.

invariance_loss: now utilizes a Scaled Cosine Binary Cross-Entropy (BCE) approach.
  - Computes cosine similarity for both same-meaning and diff-meaning pairs.
  - Applies a learnable scale and bias to these similarities to generate logits.
  - Uses BCE to push same-meaning pairs toward 1 and diff-meaning pairs toward 0.
  - This removes the need for a rigid hyperparameter margin, as the model learns 
    the optimal boundary dynamically during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import cosine_distance, normalized_l2


class ScaledCosineBCELoss(nn.Module):
    def __init__(self, initial_scale=10.0, initial_bias=0.0, max_scale=50.0):
        super().__init__()
        # Learnable parameters for dynamic margin scaling
        self.scale = nn.Parameter(torch.tensor([initial_scale]))
        self.bias = nn.Parameter(torch.tensor([initial_bias]))
        self.max_scale = max_scale
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, sim_same, sim_diff):
        # Concatenate similarities into a single batch
        sims = torch.cat([sim_same, sim_diff], dim=0)
        
        # Create labels: 1.0 for same-meaning, 0.0 for diff-meaning
        labels = torch.cat([
            torch.ones_like(sim_same),
            torch.zeros_like(sim_diff)
        ], dim=0)
        
        # Scale and shift to create logits. scale is clamped, not used raw: an
        # unconstrained learnable scale on a bounded cosine similarity has a
        # trivial way to shrink BCE loss that has nothing to do with the model
        # actually separating same/diff pairs any further -- just keep growing
        # the scale, which sharpens the sigmoid around whatever separation
        # already exists. That inflates the gradient this loss sends back into
        # the model (observed directly: gradnorm_inv grew past 1000 by the end
        # of the least stable gpt2-small/pythia-70m-deduped sae_reg run) without
        # the underlying representations needing to change at all. Clamping
        # bounds how much "free" loss reduction the scale alone can buy;
        # min=1.0 keeps a degenerate near-zero scale (which would flatten the
        # loss's gradient w.r.t. the model to ~0, the opposite failure mode)
        # off the table too. Gradient is exactly zero for the raw parameter
        # once it's pushed past either bound, so it self-limits rather than
        # needing a separate check elsewhere.
        scale = self.scale.clamp(min=1.0, max=self.max_scale)
        logits = sims * scale + self.bias
        
        # Calculate BCE
        loss = self.bce(logits, labels)
        
        return loss


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
    device: str,
    space: str,  # "raw" or "sae"
    use_support_term: bool,
    use_rank_term: bool,
    use_bce_term: bool,
    bce_criterion: ScaledCosineBCELoss,
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

    # Calculate cosine similarities
    sim_same = F.cosine_similarity(same_a, same_b)
    sim_diff = F.cosine_similarity(diff_a, diff_b)

    components = dict(
        attractive=sim_same.mean().item(), 
        repulsive=sim_diff.mean().item()
    )

    total = torch.tensor(0.0, device=device)

    if use_bce_term:
        # Compute Scaled Cosine BCE
        bce_loss = bce_criterion(sim_same, sim_diff)
        total = total + bce_loss
        components["bce_loss"] = bce_loss.item()
    
    if use_rank_term:
        rank_loss = (sim_diff[:, None] - sim_same[None, :]).mean()
        total = total + rank_loss
        components["rank_loss"] = rank_loss.item()

    if space == "sae" and use_support_term:
        support = soft_support_distance(same_a, same_b).mean()
        total = total + support
        components["support"] = support.item()

    return total, components
