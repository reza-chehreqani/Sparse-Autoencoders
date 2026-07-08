"""
Captures per-token residual-stream activations via plain PyTorch forward hooks
on the underlying HuggingFace model's transformer blocks -- NOT TransformerLens.

Step 1 used TransformerLens because its named hooks came for free in an
inference-only setting. Step 2 trains with LoRA via `peft`, which is built
around plain `transformers` module structure; mixing in TransformerLens here
would fight the tooling rather than use it (and, as of this writing, gemma3
isn't even natively supported by TransformerLens's standard HookedTransformer
-- it would need a separate, newer bridge component). So this re-implements
just the specific hook points each model's SAE was actually trained on,
directly on the HF module tree:

  - gpt2-small ("pre" / hook_resid_pre): a forward PRE-hook on transformer.h[l]
    captures the block's INPUT -- the residual stream state before block l
    runs. GPT2Model.forward calls each block as `block(hidden_states, ...)`
    with hidden_states positional, so a pre-hook's `args[0]` is exactly that.
  - pythia-70m-deduped ("post" / hook_resid_post): a forward hook on
    gpt_neox.layers[l] captures the block's OUTPUT -- the residual stream state
    after block l runs. GPTNeoXLayer may return a tuple; the hidden states are
    the first element.
  - gemma3-270m ("post" / matching Gemma Scope 2's residual-stream site): a
    forward hook on model.layers[l], same reasoning as pythia. Gemma3's
    decoder layer module path (model.model.layers) follows the same
    Llama-style convention used across most current open decoder-only models
    in HF transformers.

This module provides two activation-extraction functions with deliberately
different tradeoffs:

  - get_multi_layer_activations: unbatched (one forward pass per sentence),
    unpooled, gradient-compatible (no torch.no_grad() anywhere). Used by
    train.py, where batch sizes per step are small and gradients must flow
    from the invariance loss back into the LoRA parameters.
  - get_pooled_activations_for_eval: batched (padded, attention_mask-aware),
    pooled, and wrapped in torch.no_grad() throughout. Used by evaluate.py,
    where the test split is large enough that unbatched, gradient-tracked
    forward passes run out of memory well before finishing -- see that
    function's docstring for the full reasoning.
"""

from typing import Sequence

import torch


def get_transformer_layers(hf_model, model_family: str):
    """`hf_model` should be the plain underlying HF model (e.g. from
    `peft_model.get_base_model()`), not the PeftModel wrapper."""
    if model_family == "gpt2":
        return hf_model.transformer.h
    elif model_family == "gptneox":
        return hf_model.gpt_neox.layers
    elif model_family == "gemma3":
        return hf_model.model.layers
    else:
        raise ValueError(f"Unknown model_family: {model_family}")


class ResidualStreamRecorder:
    """Registers one hook on one layer and stores whatever it last captured."""

    def __init__(self, layer_module: torch.nn.Module, hook_side: str):
        self.captured: torch.Tensor | None = None
        if hook_side == "pre":
            self._handle = layer_module.register_forward_pre_hook(self._pre_hook)
        elif hook_side == "post":
            self._handle = layer_module.register_forward_hook(self._post_hook)
        else:
            raise ValueError(f"hook_side must be 'pre' or 'post', got {hook_side!r}")

    def _pre_hook(self, module, args):
        self.captured = args[0]

    def _post_hook(self, module, args, output):
        self.captured = output[0] if isinstance(output, tuple) else output

    def remove(self):
        self._handle.remove()


def get_multi_layer_activations(
    hf_model,
    tokenizer,
    sentences: Sequence[str],
    layers: Sequence[int],
    model_family: str,
    hook_side: str,
    device: str,
) -> dict[int, list[torch.Tensor]]:
    """
    One forward pass per sentence, capturing every layer in `layers`
    simultaneously (hooks on all requested layers are registered before the
    loop, not re-registered per layer) -- mirrors Step 1's activation_cache.py
    fix of caching every layer from a single forward pass rather than one pass
    per layer.

    Returns {layer: [per-sentence [seq_len, d_model] tensors]}, with gradients
    attached if the caller is not inside a no_grad() context.
    """
    transformer_layers = get_transformer_layers(hf_model, model_family)
    recorders = {layer: ResidualStreamRecorder(transformer_layers[layer], hook_side) for layer in layers}
    try:
        results: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        for sentence in sentences:
            tokens = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)
            hf_model(tokens)  # forward pass; we only need the recorders' side effects
            for layer in layers:
                results[layer].append(recorders[layer].captured[0])  # drop batch dim -> [seq_len, d_model]
        return results
    finally:
        for r in recorders.values():
            r.remove()
