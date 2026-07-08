"""
Re-runs Step 1's exact measurement pipeline (same metrics, same PAWS test split,
same encode-then-pool-then-compare protocol) on either the pretrained baseline
or a trained LoRA checkpoint, across every layer. This is what makes the
before/after comparison meaningful: the identical measurement applied to data
the model never saw during training (the test split), before and after training
changes anything.

Usage:
    python evaluate.py --model gpt2-small --run_id baseline
    python evaluate.py --model gpt2-small --run_id gpt2-small__C3_sae_magnitude__lam1.0 \
        --adapter_path results/step2_llm/gpt2-small__C3_sae_magnitude__lam1.0/adapter
"""

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch
from tqdm import tqdm

from config import MODEL_CONFIGS, PAWS_CONFIG, TRAIN_CONFIG
from hf_model_loading import load_model_for_eval, load_tokenizer, resolve_device
from hooked_activations import get_multi_layer_activations
from metrics import (
    LayerStats,
    cosine_distance,
    discrimination_auroc,
    latent_flip_concentration,
    support_jaccard_distance,
)
from pairs_dataset import load_paws_split
from frozen_sae import load_all_layer_saes


@torch.no_grad()
def pool_sae_codes(sae, per_sentence_acts: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([sae.encode(a).mean(dim=0) for a in per_sentence_acts], dim=0)


@torch.no_grad()
def pool_raw_activations(per_sentence_acts: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([a.mean(dim=0) for a in per_sentence_acts], dim=0)


def evaluate_checkpoint(model_key: str, run_id: str, adapter_path: str | None) -> None:
    model_cfg = MODEL_CONFIGS[model_key]
    device = resolve_device(TRAIN_CONFIG["device"])
    tau = TRAIN_CONFIG["support_threshold"]
    eval_batch_size = TRAIN_CONFIG["eval_batch_size"]
    all_layers = list(range(model_cfg.n_layers))

    print(f"[{model_key}] loading SAEs ...")
    saes = load_all_layer_saes(model_cfg, device)

    print(f"[{model_key}] loading model (adapter={adapter_path or 'none -- pretrained baseline'}) ...")
    tokenizer = load_tokenizer(model_cfg)
    _, hf_model = load_model_for_eval(model_cfg, device, adapter_path)

    print(f"[{model_key}] loading PAWS test pairs (identical split/sampling to Step 1) ...")
    test_pairs = load_paws_split(
        hf_name=PAWS_CONFIG["hf_name"],
        hf_config=PAWS_CONFIG["hf_config"],
        split="test",
        n_pairs_per_condition=PAWS_CONFIG["n_test_pairs_per_condition"],
        random_seed=PAWS_CONFIG["random_seed"],
    )
    same_pairs = [p for p in test_pairs if p.same_meaning]
    diff_pairs = [p for p in test_pairs if not p.same_meaning]

    print(f"[{model_key}] computing activations ...")
    def pooled(sentences: list[str]):
        raw_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in all_layers}
        sae_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in all_layers}
        for start in tqdm(range(0, len(sentences), eval_batch_size), desc="Pooling sentences", total=len(sentences) // eval_batch_size):
            batch = list(sentences[start : start + eval_batch_size])
            with torch.no_grad():
                acts = get_multi_layer_activations(
                    hf_model, tokenizer, batch, all_layers, model_cfg.model_family, model_cfg.hook_side, device
                )
            for layer in all_layers:
                raw_chunks[layer].append(pool_raw_activations(acts[layer]).cpu())
                sae_chunks[layer].append(pool_sae_codes(saes[layer], acts[layer]).cpu())

        pooled_raw = {layer: torch.cat(chunks, dim=0) for layer, chunks in raw_chunks.items()}
        pooled_sae = {layer: torch.cat(chunks, dim=0) for layer, chunks in sae_chunks.items()}
        return pooled_raw, pooled_sae

    raw_a_same, sae_a_same = pooled([p.sentence_a for p in same_pairs])
    raw_b_same, sae_b_same = pooled([p.sentence_b for p in same_pairs])
    raw_a_diff, sae_a_diff = pooled([p.sentence_a for p in diff_pairs])
    raw_b_diff, sae_b_diff = pooled([p.sentence_b for p in diff_pairs])

    all_layer_stats = []
    per_layer_raw_records = []

    for layer in all_layers:
        raw_cos_same = cosine_distance(raw_a_same[layer], raw_b_same[layer]).float().numpy()
        raw_cos_diff = cosine_distance(raw_a_diff[layer], raw_b_diff[layer]).float().numpy()
        sae_cos_same = cosine_distance(sae_a_same[layer], sae_b_same[layer]).float().numpy()
        sae_cos_diff = cosine_distance(sae_a_diff[layer], sae_b_diff[layer]).float().numpy()
        sae_jacc_same = support_jaccard_distance(sae_a_same[layer], sae_b_same[layer], tau).float().numpy()
        sae_jacc_diff = support_jaccard_distance(sae_a_diff[layer], sae_b_diff[layer], tau).float().numpy()
        support_diff_matrix = ((sae_a_diff[layer] > tau) ^ (sae_b_diff[layer] > tau)).float().numpy()

        auroc_raw, p_raw = discrimination_auroc(raw_cos_same, raw_cos_diff)
        auroc_sae, p_sae = discrimination_auroc(sae_cos_same, sae_cos_diff)
        top5pct_share = latent_flip_concentration(support_diff_matrix)

        stats = LayerStats(
            layer=layer,
            auroc_raw=auroc_raw,
            auroc_sae=auroc_sae,
            delta_auroc=auroc_sae - auroc_raw,
            mannwhitney_p_raw=p_raw,
            mannwhitney_p_sae=p_sae,
            top5pct_latent_share=top5pct_share,
        )
        all_layer_stats.append(stats)
        per_layer_raw_records.append(
            dict(
                layer=layer,
                same_meaning=dict(
                    raw_cos=raw_cos_same.tolist(),
                    sae_cos=sae_cos_same.tolist(),
                    sae_jacc=sae_jacc_same.tolist(),
                    token_jaccard=[p.token_jaccard for p in same_pairs],
                ),
                diff_meaning=dict(
                    raw_cos=raw_cos_diff.tolist(),
                    sae_cos=sae_cos_diff.tolist(),
                    sae_jacc=sae_jacc_diff.tolist(),
                    token_jaccard=[p.token_jaccard for p in diff_pairs],
                ),
            )
        )
        print(f"[{model_key}] layer {layer}: AUROC raw={auroc_raw:.3f}  AUROC sae={auroc_sae:.3f}  delta={stats.delta_auroc:+.3f}")

    out_dir = os.path.join(TRAIN_CONFIG["output_dir"], run_id, "eval")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{model_key}_layer_stats.json"), "w") as f:
        json.dump([asdict(s) for s in all_layer_stats], f, indent=2)
    with open(os.path.join(out_dir, f"{model_key}_raw_records.json"), "w") as f:
        json.dump(per_layer_raw_records, f)
    print(f"[{model_key}] eval results written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--run_id", required=True, help="Used to name the output directory, e.g. 'baseline' or a training run_id.")
    parser.add_argument("--adapter_path", default=None, help="Path to a saved LoRA adapter; omit for the pretrained baseline.")
    args = parser.parse_args()
    evaluate_checkpoint(args.model, args.run_id, args.adapter_path)
