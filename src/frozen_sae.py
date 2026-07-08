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
When doing this, always pair it with reconstruction_loss() as an anchor term
in the total loss. Without that anchor, a jointly-trained SAE has a trivial
way to "solve" the invariance loss: collapse to an input-independent encoding
(perfect invariance, zero information content), since nothing else in the
invariance loss cares whether the SAE still reconstructs anything. The
reconstruction term is what keeps that degenerate solution more costly than
actually satisfying the invariance objective honestly.
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
        docstring -- always pair with reconstruction_loss() when doing this."""
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

    def reconstruction_loss(self, acts: torch.Tensor) -> torch.Tensor:
        """Standard SAE training objective: reconstruction MSE between the
        input activations and the SAE's decode(encode(input)). Deliberately no
        explicit sparsity penalty on top -- SAE architectures differ in how
        they enforce sparsity, and the releases used in this project are
        TopK/JumpReLU-style, which enforce it architecturally rather than via
        an L1 term; adding one tuned for the wrong architecture could do more
        harm than good. If you know your SAE's architecture and want a
        sparsity term too, this is the place to add it.
        """
        codes = self.encode(acts)
        recon = self.decode(codes)
        return torch.nn.functional.mse_loss(recon, acts)

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
