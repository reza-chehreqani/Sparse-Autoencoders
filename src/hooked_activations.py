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


def get_trunk(hf_model, model_family: str):
    """The transformer trunk only, without the LM head. Used by the batched
    eval path to avoid computing (and allocating memory for) vocabulary-sized
    logits that aren't needed just to read intermediate activations off hooks
    -- for a short PAWS sentence this is a meaningful, easily-avoided amount of
    memory across a large batch. This matters even more for gemma3-270m, whose
    vocabulary (256k tokens) is roughly 5x GPT-2's/Pythia's."""
    if model_family == "gpt2":
        return hf_model.transformer
    elif model_family == "gptneox":
        return hf_model.gpt_neox
    elif model_family == "gemma3":
        return hf_model.model
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


def capture_layer_activations_during(hf_model, layers: Sequence[int], model_family: str, hook_side: str, forward_call):
    """
    Registers recorders on `layers`, calls `forward_call()` (a zero-arg
    callable expected to trigger a forward pass through hf_model -- e.g. via
    a wrapping PeftModel's __call__), and returns
    `(forward_call's return value, {layer: captured_tensor})`.

    This exists so a forward pass that's already happening for another reason
    -- the LM loss's forward pass over a WikiText batch, in train.py -- can
    also yield the invariance layer(s)' activations on that same batch,
    without a second forward pass. Used to extend the SAE reconstruction/
    sparsity anchor terms (--joint_sae, --sae_reg) to cover the general-text
    distribution the LM loss actually trains on, not only the narrower PAWS
    sentence distribution the invariance loss uses -- the model's activations
    can drift on the former even while the latter looks fine, since the LM
    loss pushes on every step and the invariance loss's anchor previously
    never saw that distribution at all.

    Captured tensors keep whatever shape the hook naturally produces (e.g.
    [batch, seq_len, d_model] for a padded/fixed-length batch) -- unlike
    get_multi_layer_activations, which returns per-sentence lists from
    unbatched single-sentence forward passes.
    """
    transformer_layers = get_transformer_layers(hf_model, model_family)
    recorders = {layer: ResidualStreamRecorder(transformer_layers[layer], hook_side) for layer in layers}
    try:
        result = forward_call()
        captured = {layer: recorders[layer].captured for layer in layers}
        return result, captured
    finally:
        for r in recorders.values():
            r.remove()


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
    TRAINING path. One forward pass per sentence, capturing every layer in
    `layers` simultaneously (hooks for all requested layers are registered
    before the loop, not re-registered per layer). Returns
    {layer: [per-sentence [seq_len, d_model] tensors]}, with gradients attached
    if the caller is not inside a no_grad() context -- deliberately not wrapped
    in no_grad() here; see module docstring.
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


@torch.no_grad()
def get_pooled_activations_for_eval(
    hf_model,
    tokenizer,
    sentences: Sequence[str],
    layers: Sequence[int],
    saes: dict,
    model_family: str,
    hook_side: str,
    device: str,
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """
    EVALUATION path. Fixes three things that don't matter at training's small
    per-step batch sizes but do at evaluation scale (a full PAWS test split):

      1. Every layer is captured from the SAME set of forward passes -- hooks
         for all of `layers` are registered once, up front -- rather than
         re-running the model once per layer.
      2. Sentences are tokenized and run in padded batches of `batch_size`
         (attention_mask-aware), not one sentence at a time.
      3. The whole function runs under torch.no_grad(), and only the
         transformer trunk is called (not the full LM head), so no autograd
         graph and no unused vocabulary-sized logit tensor is ever built or
         retained. This is the fix that actually matters most: without it,
         every one of the (sentence x layer) forward passes retains a full
         graph back through the model, and those retained graphs accumulating
         over a full test split is exactly what runs out of memory, even
         though no single forward pass is large on its own.

    Encoding order still follows the encode-per-token-then-pool rule used
    everywhere else in this project: the SAE encodes every position, including
    padding positions (harmless -- they're masked out immediately after), and
    pooling happens on the resulting per-token codes, not on a pre-averaged raw
    vector.

    Returns (pooled_raw, pooled_sae), each {layer: [n_sentences, dim]}. Results
    are moved to CPU immediately after each batch, so accumulated memory lives
    in system RAM rather than holding GPU memory for the whole dataset at once.
    """
    trunk = get_trunk(hf_model, model_family)
    transformer_layers = get_transformer_layers(hf_model, model_family)
    recorders = {layer: ResidualStreamRecorder(transformer_layers[layer], hook_side) for layer in layers}

    raw_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    sae_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    try:
        for start in range(0, len(sentences), batch_size):
            batch = list(sentences[start : start + batch_size])
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            trunk(input_ids, attention_mask=attention_mask)

            mask = attention_mask.unsqueeze(-1).float()       # [b, seq_len, 1]
            denom = mask.sum(dim=1).clamp_min(1.0)               # [b, 1]
            for layer in layers:
                acts = recorders[layer].captured                  # [b, seq_len, d_model]
                raw_chunks[layer].append(((acts * mask).sum(dim=1) / denom).cpu())

                codes = saes[layer].encode(acts)                     # [b, seq_len, d_sae]
                sae_chunks[layer].append(((codes * mask).sum(dim=1) / denom).cpu())

        pooled_raw = {layer: torch.cat(chunks, dim=0) for layer, chunks in raw_chunks.items()}
        pooled_sae = {layer: torch.cat(chunks, dim=0) for layer, chunks in sae_chunks.items()}
        return pooled_raw, pooled_sae
    finally:
        for r in recorders.values():
            r.remove()
