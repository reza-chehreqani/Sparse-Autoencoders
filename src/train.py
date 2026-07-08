"""
Trains one (model, condition, lambda) combination for Step 2.

Each step draws one WikiText batch and (for C2-C4) one PAWS same-meaning batch,
and combines both into a single backward pass:

    total_loss = lm_loss(wikitext_batch) + lambda * invariance_loss(paws_batch)

Produces, under results/step2_llm/<model>__<condition>__lam<lambda>/:
  - adapter/            the trained LoRA adapter (not the full model)
  - training_log.json   periodic validation checks: LM perplexity, SAE-space
                         same/diff AUROC (the collapse check -- this should
                         WIDEN over training, not shrink), mean same/diff
                         SAE-cosine-distance, and SAE reconstruction variance
                         explained (the frozen-SAE-drift check).

Usage:
    python train.py --model gpt2-small --condition C3_sae_magnitude --lam 1.0
"""

import argparse
import json
import os
import random

import torch
from tqdm import tqdm

from config import CONDITION_SPECS, INVARIANCE_LAYERS, MODEL_CONFIGS, PAWS_CONFIG, TRAIN_CONFIG, WIKITEXT_CONFIG
from frozen_sae import load_layer_saes
from hf_model_loading import load_lora_model, load_tokenizer, resolve_device
from hooked_activations import get_multi_layer_activations
from lm_dataset import WikiTextBatcher, load_wikitext_train_text
from losses import invariance_loss, lm_loss, pool_sae_codes_for_training
from metrics import cosine_distance, discrimination_auroc
from pairs_dataset import load_paws_split


def _paws_split_kwargs(split: str) -> dict:
    n_key = {
        "train": "n_train_pairs_per_condition",
        "validation": "n_validation_pairs_per_condition",
        "test": "n_test_pairs_per_condition",
    }[split]
    return dict(
        hf_name=PAWS_CONFIG["hf_name"],
        hf_config=PAWS_CONFIG["hf_config"],
        split=split,
        n_pairs_per_condition=PAWS_CONFIG[n_key],
        random_seed=PAWS_CONFIG["random_seed"],
    )


def sample_batch(pairs: list, batch_size: int, rng_state: dict, key: str) -> list:
    """Cycles through `pairs` without replacement within an epoch, reshuffling at
    wraparound. `rng_state[key]` tracks position + shuffled order independently
    per call site."""
    if key not in rng_state:
        order = list(range(len(pairs)))
        random.Random(0).shuffle(order)
        rng_state[key] = dict(order=order, pos=0)
    state = rng_state[key]
    if state["pos"] + batch_size > len(state["order"]):
        random.Random(state["pos"]).shuffle(state["order"])
        state["pos"] = 0
    idx = state["order"][state["pos"] : state["pos"] + batch_size]
    state["pos"] += batch_size
    return [pairs[i] for i in idx]


@torch.no_grad()
def run_validation_checks(
    peft_model,
    hf_model,
    tokenizer,
    saes,
    inv_layers,
    model_family,
    hook_side,
    val_same_pairs,
    val_diff_pairs,
    wikitext_val_batcher,
    device,
) -> dict:
    peft_model.eval()

    val_batch = wikitext_val_batcher.next_batch(8)
    ppl = torch.exp(lm_loss(peft_model, val_batch)).item()

    layer = inv_layers[0]  # monitor the first invariance layer; training may use more than one
    sentences = dict(
        a_same=[p.sentence_a for p in val_same_pairs],
        b_same=[p.sentence_b for p in val_same_pairs],
        a_diff=[p.sentence_a for p in val_diff_pairs],
        b_diff=[p.sentence_b for p in val_diff_pairs],
    )
    acts = {
        k: get_multi_layer_activations(hf_model, tokenizer, v, [layer], model_family, hook_side, device)[layer]
        for k, v in sentences.items()
    }

    sae = saes[layer]
    same_cos = (
        cosine_distance(pool_sae_codes_for_training(sae, acts["a_same"]), pool_sae_codes_for_training(sae, acts["b_same"]))
        .cpu()
        .float()
        .numpy()
    )
    diff_cos = (
        cosine_distance(pool_sae_codes_for_training(sae, acts["a_diff"]), pool_sae_codes_for_training(sae, acts["b_diff"]))
        .cpu()
        .float()
        .numpy()
    )
    auroc_sae, _ = discrimination_auroc(same_cos, diff_cos)

    all_same_tokens = torch.cat(acts["a_same"] + acts["b_same"], dim=0)
    var_explained = sae.reconstruction_variance_explained(all_same_tokens)

    peft_model.train()
    return dict(
        perplexity=ppl,
        auroc_sae=float(auroc_sae),
        mean_same_sae_cos=float(same_cos.mean()),
        mean_diff_sae_cos=float(diff_cos.mean()),
        sae_variance_explained=float(var_explained),
    )


def run(model_key: str, condition: str, lam: float, seed: int = 0) -> None:
    torch.manual_seed(seed)
    model_cfg = MODEL_CONFIGS[model_key]
    spec = CONDITION_SPECS[condition]
    inv_layers = INVARIANCE_LAYERS[model_key]
    device = resolve_device(TRAIN_CONFIG["device"])

    print(f"[{model_key}/{condition}/lam={lam}] loading SAEs ...")
    saes = load_layer_saes(model_cfg, device, inv_layers)

    print(f"[{model_key}/{condition}/lam={lam}] loading LoRA model ...")
    tokenizer = load_tokenizer(model_cfg)
    peft_model = load_lora_model(model_cfg, TRAIN_CONFIG, device)
    hf_model = peft_model.get_base_model()

    print(f"[{model_key}/{condition}/lam={lam}] loading data ...")
    train_pairs = load_paws_split(**_paws_split_kwargs("train"))
    train_same = [p for p in train_pairs if p.same_meaning]
    val_pairs = load_paws_split(**_paws_split_kwargs("validation"))
    val_same = [p for p in val_pairs if p.same_meaning]
    val_diff = [p for p in val_pairs if not p.same_meaning]

    wikitext_texts = load_wikitext_train_text(WIKITEXT_CONFIG["hf_name"], WIKITEXT_CONFIG["hf_config"])
    wikitext_train_batcher = WikiTextBatcher(
        wikitext_texts[:-200], tokenizer, WIKITEXT_CONFIG["max_seq_len"], device, seed=seed
    )
    wikitext_val_batcher = WikiTextBatcher(
        wikitext_texts[-200:], tokenizer, WIKITEXT_CONFIG["max_seq_len"], device, seed=seed + 1
    )

    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=TRAIN_CONFIG["learning_rate"])

    rng_state: dict = {}
    log = []
    for step in tqdm(range(TRAIN_CONFIG["max_steps"]), desc="Training steps", total=TRAIN_CONFIG["max_steps"]):
        lm_batch = wikitext_train_batcher.next_batch(TRAIN_CONFIG["batch_size_lm"])
        loss_lm = lm_loss(peft_model, lm_batch)

        if spec["use_invariance"]:
            same_batch = sample_batch(train_same, TRAIN_CONFIG["batch_size_invariance"], rng_state, "train_same")
            sentences_a = [p.sentence_a for p in same_batch]
            sentences_b = [p.sentence_b for p in same_batch]
            acts_a = get_multi_layer_activations(
                hf_model, tokenizer, sentences_a, inv_layers, model_cfg.model_family, model_cfg.hook_side, device
            )
            acts_b = get_multi_layer_activations(
                hf_model, tokenizer, sentences_b, inv_layers, model_cfg.model_family, model_cfg.hook_side, device
            )
            loss_inv = sum(
                invariance_loss(saes[l], acts_a[l], acts_b[l], spec["space"], spec["use_support_term"])
                for l in inv_layers
            )
            total_loss = loss_lm + lam * loss_inv
        else:
            loss_inv = torch.tensor(0.0)
            total_loss = loss_lm

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % TRAIN_CONFIG["eval_every"] == 0 or step == TRAIN_CONFIG["max_steps"] - 1:
            checks = run_validation_checks(
                peft_model, hf_model, tokenizer, saes, inv_layers, model_cfg.model_family, model_cfg.hook_side,
                val_same, val_diff, wikitext_val_batcher, device,
            )
            checks.update(step=step, loss_lm=float(loss_lm.item()), loss_inv=float(loss_inv.item()))
            log.append(checks)
            print(
                f"  step {step}: lm_loss={checks['loss_lm']:.3f} inv_loss={checks['loss_inv']:.4f} "
                f"ppl={checks['perplexity']:.2f} auroc_sae={checks['auroc_sae']:.3f} "
                f"same_cos={checks['mean_same_sae_cos']:.3f} diff_cos={checks['mean_diff_sae_cos']:.3f} "
                f"sae_var_explained={checks['sae_variance_explained']:.3f}"
            )

    run_id = f"{model_key}__{condition}__lam{lam}"
    out_dir = os.path.join(TRAIN_CONFIG["output_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(os.path.join(out_dir, "adapter"))
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[{run_id}] done, saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--condition", required=True, choices=list(CONDITION_SPECS.keys()))
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.model, args.condition, args.lam, args.seed)
