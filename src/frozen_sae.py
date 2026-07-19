"""
SAELens SAE wrapper for TRAINING use. Loads the exact same pretrained SAE as
Step 1 (via SAE.from_pretrained), but deliberately WITHOUT the @torch.no_grad()
decorators Step 1's inference-only wrapper used on encode()/decode() -- those
decorators would silently break gradient flow here.

Frozen by default. The SAE's own parameters start with requires_grad_(False),
which is the correct and sufficient way to keep it un-trained while still
letting gradients flow *through* it into the LoRA-adapted model: PyTorch
autograd still computes gradients *with respect to the input* of a frozen
module's forward pass (this is exactly what makes, e.g., a frozen VGG usable
inside a perceptual loss) -- it simply never accumulates a gradient *into*
that module's own (non-trainable) parameters.

Can be made trainable via make_trainable(), for train.py's --joint_sae flag.
When doing this, always pair it with training_losses() as an anchor in the
total loss. Without that anchor, a jointly-trained SAE has a trivial way to
"solve" the invariance loss: collapse to an input-independent encoding
(perfect invariance, zero information content), since nothing else in the
invariance loss cares whether the SAE still reconstructs anything or stays
sparse. The reconstruction term is what keeps that degenerate solution more
costly than actually satisfying the invariance objective honestly; the
sparsity term is what keeps the SAE from just getting denser and denser as an
easy way to reduce reconstruction error, defeating the point of using an SAE
at all.
"""

import warnings
import torch
from sae_lens import SAE


class FrozenSAE:
    def __init__(self, release: str, sae_id: str, device: str, d_in: int):
        self._sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        for p in self._sae.parameters():
            p.requires_grad_(False)
        self._sae.eval()
        self.d_in = d_in
        self.trainable = False

        self.model_from_pretrained_kwargs = dict(getattr(getattr(self._sae.cfg, "metadata", None), "model_from_pretrained_kwargs", {}) or {})
        self._should_center = self.model_from_pretrained_kwargs.get("center_writing_weights", False)

        if self.model_from_pretrained_kwargs and not self._should_center:
            warnings.warn(f"Unhandled kwargs in SAE {sae_id}: {self.model_from_pretrained_kwargs}")

    def make_trainable(self) -> None:
        """Unfreezes this SAE's parameters for joint training. See module
        docstring -- always pair with training_losses() when doing this."""
        for p in self._sae.parameters():
            p.requires_grad_(True)
        self._sae.train()
        self.trainable = True

    def parameters(self):
        return self._sae.parameters()

    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        if self._should_center:
            acts = acts - acts.mean(dim=-1, keepdim=True)
        return self._sae.encode(acts)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self._sae.decode(codes)

    def training_losses(self, acts: torch.Tensor) -> dict:
        """The SAE's own standard training objective, for --joint_sae's anchor
        term: reconstruction MSE plus an L1 sparsity penalty on the code. Both
        come from a single encode() call rather than two, to avoid doubling
        the forward-pass cost.

        The L1 term is deliberately a generic, architecture-agnostic choice:
        codes are non-negative (ReLU-based encoders), so their L1 norm is just
        their sum. This is the traditional sparsity loss for "standard"/
        "gated" SAEs and is always well-defined regardless of architecture,
        but it is NOT necessarily identical to how every SAE release used in
        this project was originally trained -- TopK/JumpReLU SAEs (several of
        the releases here) enforce sparsity architecturally (a hard top-k
        selection, or a learned per-feature threshold) rather than via L1, so
        this term acts as an additional soft regularizer for those rather than
        a reproduction of their native training loss. If you know a given
        release's exact architecture (`sae.cfg.architecture` reports it) and
        want to match its native objective exactly -- e.g. an L0-style penalty
        for JumpReLU -- this is the place to specialize it.

        Raw L1 magnitude scales with dictionary width and is typically much
        larger than the reconstruction MSE, so it needs a small weight
        (TRAIN_CONFIG's sparsity_loss_weight) to avoid dominating the total
        loss and crushing the SAE's activity toward zero -- watch mean_l0()
        below alongside reconstruction_variance_explained() to check the
        weight is in a sane range, not just the loss value going down.

        RECONSTRUCTION IS AN AGGREGATE (SUM/SUM) RATIO, NOT A PER-TOKEN
        AVERAGE. This replaced a plain torch.nn.functional.mse_loss(recon,
        acts) after the first full Step-2 ablation on gpt2-small showed
        --sae_reg's reconstruction term (train.py's loss_inv_sae /
        loss_lm_sae) making up 70-97% of total_loss at lambda>=1, with the
        actual invariance loss essentially flat/unoptimized as a result. That
        first fix computed a PER-TOKEN ratio and averaged the ratios:

            per_token_sq_err / per_token_sq_norm.clamp_min(1e-6), then .mean()

        which fixed the original scale/volatility problem but introduced a
        new, worse one: user-run diagnostics (a clean A/B -- identical model,
        seed, WikiText batch order, and lambda=0 for C1_lm_only, differing
        ONLY in --sae_reg) isolated a step-50 perplexity spike to 100M+ that
        occurred ONLY when --sae_reg was on, never otherwise, on an otherwise
        completely deterministic WikiText batch order. Mechanism: if even one
        token in a batch has a near-zero activation norm at the invariance
        layer, clamp_min(1e-6) does not stop per_token_sq_err /
        per_token_sq_norm from being enormous for that one token -- and
        because the old formula AVERAGES ratios rather than summing before
        dividing, one such token is a lone, unbounded term inside a mean, not
        a term that gets diluted by the batch. Weighted by
        sae_recon_loss_weight=50 and backpropagated with no gradient clipping
        anywhere in train.py (added below, but this fix removes the root
        cause rather than just cushioning it), that one token was enough to
        wreck the model for a step (observed directly: lm_loss went from
        11.568 at init to 12.684 -- WORSE than initialization -- at the exact
        step this fired, while an identical run without --sae_reg had already
        dropped to 5.955 on the same data).

        The aggregate form below computes ONE ratio for the whole batch --
        sum of squared errors over sum of squared norms -- instead of
        averaging per-token ratios. A single near-zero-norm token now barely
        moves either sum, rather than standing alone as an unbounded
        division. This is the same "fraction of activation energy left
        unreconstructed" quantity as the reconstruction_variance_explained()
        diagnostic already uses below, just applied inside the loss itself
        instead of only for logging -- and it's still the same order of
        magnitude fix relative to raw MSE that motivated normalizing in the
        first place (comparable scale to the invariance loss, not swamping
        it), without the per-token-average's fragility to any single outlier.
        The explicit .clamp(max=...) below is deliberate defense-in-depth on
        top of the structural fix, not a substitute for it -- it bounds the
        worst case if some future SAE/activation combination still produces
        a pathological ratio for a reason this fix doesn't anticipate.

        NOTE: this changes the numeric scale of TRAIN_CONFIG's
        sae_recon_loss_weight -- it needs to be re-tuned, not left at
        whatever value worked against raw MSE. See config.py's comment on
        that value, and use train.py's new gradnorm_* diagnostics (logged
        alongside the loss components) to check the anchor's actual gradient
        contribution against the invariance loss's, rather than eyeballing
        loss magnitudes the way the old bug was originally missed.
        """
        if self._should_center:
            acts = acts - acts.mean(dim=-1, keepdim=True)

        codes = self.encode(acts)
        recon = self.decode(codes)

        per_token_sq_err = (recon - acts).pow(2).sum(dim=-1)
        per_token_sq_norm = acts.pow(2).sum(dim=-1)

        # Aggregate BEFORE dividing -- see docstring above for why this,
        # rather than per_token_sq_err/per_token_sq_norm-then-.mean(), is what
        # actually fixes the near-zero-denominator instability, not just
        # relocates it.
        reconstruction = per_token_sq_err.sum() / per_token_sq_norm.sum().clamp_min(1e-6)
        # Defense-in-depth, not the primary fix: bounds the worst case even if
        # something else unexpected produces a large aggregate ratio (e.g. an
        # entire batch of degenerate activations, which the sum/sum form alone
        # wouldn't protect against the way it protects against a single token).
        reconstruction = reconstruction.clamp(max=10.0)

        sparsity = codes.sum(dim=-1).mean()
        return dict(
            reconstruction=reconstruction,
            sparsity=sparsity,
            # Unweighted, detached, logging-only -- lets training_log.json show
            # the old-scale number too, for continuity when comparing against
            # the first ablation's logs.
            reconstruction_mse_raw=torch.nn.functional.mse_loss(recon, acts).detach(),
        )

    @torch.no_grad()
    def mean_l0(self, acts: torch.Tensor) -> float:
        """Average number of active (nonzero) latents per token -- a direct,
        interpretable sparsity readout. Useful for checking the sparsity term
        in --joint_sae training is having a sane effect: shouldn't collapse
        toward ~0 active features (too aggressive a weight) or drift toward
        counting most of the dictionary as active (too weak, or absent)."""
        codes = self.encode(acts)
        return (codes > 0).float().sum(dim=-1).mean().item()

    @torch.no_grad()
    def reconstruction_variance_explained(self, acts: torch.Tensor) -> float:
        """Fraction of activation variance the SAE's reconstruction explains,
        used only for the periodic SAE-drift check during training (never for
        the training loss itself, so no_grad here is correct and deliberate).
        As LoRA adapts the model, its activation distribution can drift away
        from what this frozen SAE was originally fit to; a falling value here
        is the direct signal that the "frozen SAE" assumption is breaking down
        and needs revisiting (e.g. shorter runs, smaller LoRA rank, or
        periodically refreshing / jointly training the SAE)."""
        if self._should_center:
            acts = acts - acts.mean(dim=-1, keepdim=True)

        codes = self.encode(acts)
        recon = self.decode(codes)
        residual_var = (acts - recon).var().item()
        total_var = acts.var().item()
        return 1.0 - residual_var / max(total_var, 1e-8)

    def save(self, path: str) -> None:
        """Saves this SAE's current weights -- meaningful to call only after
        make_trainable() plus some training steps; used by train.py's
        --joint_sae path to persist the jointly-trained SAE alongside the
        LoRA adapter."""
        torch.save(self._sae.state_dict(), path)

    def load_trained_state(self, path: str, device: str) -> None:
        """Overlays a previously-saved, jointly-trained state on top of this
        SAE's current (pretrained) weights. Used by evaluate.py to evaluate a
        checkpoint that used --joint_sae with the SAE it actually ended
        training with, rather than the fresh pretrained one."""
        state = torch.load(path, map_location=device)
        self._sae.load_state_dict(state)


def load_layer_saes(model_cfg, device: str, layers: list[int] | None = None) -> dict[int, FrozenSAE]:
    requested_layers = list(layers) if layers is not None else list(range(model_cfg.n_layers))
    saes: dict[int, FrozenSAE] = {}
    for layer in requested_layers:
        sae_id = model_cfg.sae_id_template.format(layer=layer)
        saes[layer] = FrozenSAE(model_cfg.sae.release, sae_id, device, d_in=model_cfg.d_model)
    return saes


def load_all_layer_saes(model_cfg, device: str) -> dict[int, FrozenSAE]:
    return load_layer_saes(model_cfg, device, list(range(model_cfg.n_layers)))
