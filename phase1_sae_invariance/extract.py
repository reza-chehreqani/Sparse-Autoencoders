"""
Two-stage extraction, designed to keep peak GPU memory low:

  Stage A: run every sentence through the base model ONCE, caching the
           pooled residual-stream vector at every layer of interest.
           Only the base model needs to be resident on the GPU for this.

  Stage B: for each layer, load ONLY that layer's Gemma Scope SAE,
           batch-encode every cached residual vector for that layer,
           record cosine similarities, then free the SAE before moving
           to the next layer.

This means peak GPU memory is roughly (base model) + (one SAE), not
(base model) + (all 26 SAEs) simultaneously, which matters on a single
consumer GPU.
"""

import gc
import torch
from tqdm import tqdm

from config import Phase1Config


def load_model(cfg: Phase1Config):
    # Imported here (not at module top) so the smoke test, which never
    # touches the model, doesn't require torch/transformer_lens/sae_lens
    # to be installed.
    from sae_lens import HookedSAETransformer

    model = HookedSAETransformer.from_pretrained_no_processing(cfg.model_name, device=cfg.device)
    return model


def _pool(tensor: torch.Tensor, pooling: str) -> torch.Tensor:
    # tensor: [seq, d]
    if pooling == "mean":
        return tensor.mean(dim=0)
    elif pooling == "last_token":
        return tensor[-1]
    raise ValueError(f"Unknown pooling: {pooling}")


@torch.no_grad()
def cache_residuals(model, cfg: Phase1Config, pairs):
    """
    Stage A. Returns a list of dicts, one per pair:
        {is_paraphrase, sentence_a, sentence_b,
         raw_a: {layer: pooled_residual_vector (on CPU)},
         raw_b: {layer: pooled_residual_vector (on CPU)}}
    Residuals are moved to CPU immediately so GPU memory stays free for
    Stage B's SAEs.
    """
    hook_names = [cfg.hook_point_template.format(layer=l) for l in cfg.layers]
    cached = []

    for pair in tqdm(pairs, desc="[Stage A] Caching residual streams"):
        entry = {
            "is_paraphrase": pair["is_paraphrase"],
            "sentence_a": pair["sentence_a"],
            "sentence_b": pair["sentence_b"],
            "raw_a": {},
            "raw_b": {},
        }

        for side, key in [("sentence_a", "raw_a"), ("sentence_b", "raw_b")]:
            tokens = model.to_tokens(pair[side], prepend_bos=True)[:, : cfg.max_seq_len]
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in hook_names)
            for layer in cfg.layers:
                resid = cache[cfg.hook_point_template.format(layer=layer)][0]  # [seq, d_model]
                entry[key][layer] = _pool(resid, cfg.pooling).float().cpu()

        cached.append(entry)

    return cached


@torch.no_grad()
def compute_sae_cosine_per_layer(cached, cfg: Phase1Config):
    """
    Stage B. For each layer: load only that layer's SAE, batch-encode
    every cached residual for that layer, compute cosine similarity,
    free the SAE, move on.

    Returns a flat list of dict rows:
        {layer, is_paraphrase, sae_cosine, raw_cosine}
    """
    from sae_lens import SAE

    rows = []
    is_paraphrase_flags = [entry["is_paraphrase"] for entry in cached]

    for layer in tqdm(cfg.layers, desc="[Stage B] Encoding layer-by-layer"):
        sae_id = f"layer_{layer}/{cfg.sae_width}/canonical"
        sae, _, _ = SAE.from_pretrained(release=cfg.sae_release, sae_id=sae_id, device=cfg.device)

        raw_a_stack = torch.stack([e["raw_a"][layer] for e in cached]).to(cfg.device)
        raw_b_stack = torch.stack([e["raw_b"][layer] for e in cached]).to(cfg.device)

        feat_a = sae.encode(raw_a_stack)  # [N, d_sae]
        feat_b = sae.encode(raw_b_stack)  # [N, d_sae]

        sae_cos = torch.nn.functional.cosine_similarity(feat_a, feat_b, dim=1).cpu().numpy()
        raw_cos = torch.nn.functional.cosine_similarity(raw_a_stack, raw_b_stack, dim=1).cpu().numpy()

        for i, is_para in enumerate(is_paraphrase_flags):
            rows.append({
                "layer": layer,
                "is_paraphrase": bool(is_para),
                "sae_cosine": float(sae_cos[i]),
                "raw_cosine": float(raw_cos[i]),
            })

        del sae, feat_a, feat_b, raw_a_stack, raw_b_stack
        gc.collect()
        torch.cuda.empty_cache()

    return rows
