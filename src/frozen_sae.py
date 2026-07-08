"""
SAELens SAE wrapper for TRAINING use. Loads the exact same pretrained SAE as
Step 1 (via SAE.from_pretrained), but deliberately WITHOUT the @torch.no_grad()
decorators Step 1's inference-only wrapper used on encode()/decode() -- those
decorators would silently break gradient flow here.

The SAE's own parameters are frozen via requires_grad_(False) at construction
time, which is the correct and sufficient way to keep it un-trained: PyTorch
autograd still computes gradients *with respect to the input* of a frozen
module's forward pass (this is exactly what makes, e.g., a frozen VGG usable
inside a perceptual loss) -- it simply never accumulates a gradient *into* that
module's own (non-trainable) parameters. So encode()/decode() below are
differentiable with respect to whatever activations are fed in, which is what
lets the invariance loss's gradient reach back into the LoRA-adapted model,
while the SAE dictionary itself never changes.
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
                
        self.model_from_pretrained_kwargs = dict(getattr(getattr(self._sae.cfg, "metadata", None), "model_from_pretrained_kwargs", {}) or {})
        self._should_center = self.model_from_pretrained_kwargs.get("center_writing_weights", False)

        if self.model_from_pretrained_kwargs and not self._should_center:
            warnings.warn(f"Unhandled kwargs in SAE {sae_id}: {self.model_from_pretrained_kwargs}")

    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        if self._should_center:
            acts = acts - acts.mean(dim=-1, keepdim=True)
        return self._sae.encode(acts)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self._sae.decode(codes)

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


def load_layer_saes(model_cfg, device: str, layers: list[int] | None = None) -> dict[int, FrozenSAE]:
    requested_layers = list(layers) if layers is not None else list(range(model_cfg.n_layers))
    saes: dict[int, FrozenSAE] = {}
    for layer in requested_layers:
        sae_id = model_cfg.sae_id_template.format(layer=layer)
        saes[layer] = FrozenSAE(model_cfg.sae.release, sae_id, device, d_in=model_cfg.d_model)
    return saes


def load_all_layer_saes(model_cfg, device: str) -> dict[int, FrozenSAE]:
    return load_layer_saes(model_cfg, device, list(range(model_cfg.n_layers)))
