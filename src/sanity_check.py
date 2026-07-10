"""
A minimal smoke test for the Step 2 training pipeline: loads one model, runs a
handful of optimizer steps for the SAE-magnitude-plus-support condition under
all three SAE-handling modes (frozen/default, --joint_sae, --sae_reg), and
checks that:
  - the loss is finite in every mode
  - at least one LoRA parameter receives a nonzero gradient in every mode,
    INCLUDING sae_reg -- this is the specific thing worth checking there,
    since the whole point of sae_reg is that gradient from a frozen SAE's
    reconstruction loss still reaches the model
  - in joint_sae mode, the SAE's own parameters receive a nonzero gradient
    AND actually change value after optimizer steps
  - in sae_reg mode, the SAE's own parameters do NOT change value at all,
    despite its reconstruction loss being part of the total loss

This catches the "forgot to unfreeze the right parameters", "SAE encode()
accidentally left under no_grad", and "sae_reg accidentally left the SAE
trainable" classes of bug before committing to a full training run.
"""

import torch

from config import INVARIANCE_LAYERS, MODEL_CONFIGS, TRAIN_CONFIG, WIKITEXT_CONFIG
from frozen_sae import load_all_layer_saes
from hf_model_loading import load_lora_model, load_tokenizer, resolve_device
from hooked_activations import get_multi_layer_activations
from lm_dataset import WikiTextBatcher, load_wikitext_train_text
from losses import invariance_loss, lm_loss

SAME_MEANING_PAIRS = [
    ("The cat sat on the mat.", "The mat had a cat sitting on it."),
    ("She quickly finished her homework.", "Her homework was finished by her quickly."),
]
DIFF_MEANING_PAIRS = [
    ("The cat sat on the mat.", "The dog ran across the yard."),
    ("She quickly finished her homework.", "He slowly forgot his keys at home."),
]


def run_steps(model_key, model_cfg, mode: str):
    assert mode in ("frozen", "joint_sae", "sae_reg")
    device = resolve_device(TRAIN_CONFIG["device"])
    inv_layers = INVARIANCE_LAYERS[model_key][:1]  # just one layer for the smoke test
    layer = inv_layers[0]

    saes = load_all_layer_saes(model_cfg, device)
    if mode == "joint_sae":
        saes[layer].make_trainable()
    # frozen and sae_reg both leave the SAE frozen; they differ only in
    # whether its reconstruction loss is added to the total loss below.

    tokenizer = load_tokenizer(model_cfg)
    peft_model = load_lora_model(model_cfg, TRAIN_CONFIG, device)
    hf_model = peft_model.get_base_model()

    wikitext_texts = load_wikitext_train_text(WIKITEXT_CONFIG["hf_name"], WIKITEXT_CONFIG["hf_config"])
    batcher = WikiTextBatcher(wikitext_texts[:50], tokenizer, WIKITEXT_CONFIG["max_seq_len"], device)

    param_groups = [dict(params=[p for p in peft_model.parameters() if p.requires_grad], lr=1e-3)]
    if mode == "joint_sae":
        param_groups.append(dict(params=list(saes[layer].parameters()), lr=1e-4))
    optimizer = torch.optim.AdamW(param_groups)

    lora_param = next(p for p in peft_model.parameters() if p.requires_grad)
    sae_param = next(saes[layer].parameters())
    sae_param_before = sae_param.detach().clone()

    losses, lora_grad_norms, sae_grad_norms = [], [], []
    for step in range(3):
        lm_batch = batcher.next_batch(2)
        same_a = get_multi_layer_activations(
            hf_model, tokenizer, [p[0] for p in SAME_MEANING_PAIRS], inv_layers,
            model_cfg.model_family, model_cfg.hook_side, device,
        )
        same_b = get_multi_layer_activations(
            hf_model, tokenizer, [p[1] for p in SAME_MEANING_PAIRS], inv_layers,
            model_cfg.model_family, model_cfg.hook_side, device,
        )
        diff_a = get_multi_layer_activations(
            hf_model, tokenizer, [p[0] for p in DIFF_MEANING_PAIRS], inv_layers,
            model_cfg.model_family, model_cfg.hook_side, device,
        )
        diff_b = get_multi_layer_activations(
            hf_model, tokenizer, [p[1] for p in DIFF_MEANING_PAIRS], inv_layers,
            model_cfg.model_family, model_cfg.hook_side, device,
        )

        inv_total, _ = invariance_loss(
            saes[layer], same_a[layer], same_b[layer], diff_a[layer], diff_b[layer],
            space="sae", use_support_term=True, repulsive_margin=TRAIN_CONFIG["repulsive_margin"],
        )
        loss = lm_loss(peft_model, lm_batch) + 1.0 * inv_total

        if mode == "joint_sae":
            all_acts = torch.cat(same_a[layer] + same_b[layer] + diff_a[layer] + diff_b[layer], dim=0)
            sae_losses = saes[layer].training_losses(all_acts)
            loss = (
                loss
                + TRAIN_CONFIG["sae_recon_loss_weight"] * sae_losses["reconstruction"]
                + TRAIN_CONFIG["sparsity_loss_weight"] * sae_losses["sparsity"]
            )
        elif mode == "sae_reg":
            all_acts = torch.cat(same_a[layer] + same_b[layer] + diff_a[layer] + diff_b[layer], dim=0)
            recon = saes[layer].training_losses(all_acts)["reconstruction"]
            loss = loss + TRAIN_CONFIG["sae_recon_loss_weight"] * recon

        optimizer.zero_grad()
        loss.backward()
        lora_grad_norms.append(lora_param.grad.norm().item() if lora_param.grad is not None else 0.0)
        if mode == "joint_sae":
            sae_grad_norms.append(sae_param.grad.norm().item() if sae_param.grad is not None else 0.0)
        optimizer.step()
        losses.append(loss.item())
        print(f"  step {step}: loss={loss.item():.4f}  lora_grad_norm={lora_grad_norms[-1]:.6f}"
              + (f"  sae_grad_norm={sae_grad_norms[-1]:.6f}" if mode == "joint_sae" else ""))

    assert all(l == l for l in losses), "Loss was NaN at some step."  # l == l is False only for NaN
    assert any(g > 0 for g in lora_grad_norms), (
        "No gradient reached the LoRA parameter across any step -- check hook wiring / no_grad usage."
    )

    if mode == "joint_sae":
        assert any(g > 0 for g in sae_grad_norms), (
            "No gradient reached the SAE's own parameters -- check make_trainable() / that it's in the optimizer."
        )
        assert not torch.equal(sae_param_before, sae_param), (
            "SAE parameter values did not change after optimizer steps despite --joint_sae -- "
            "check that its parameters were actually added to the optimizer."
        )
    else:
        # frozen and sae_reg: the SAE must come out of this bit-for-bit identical,
        # even though sae_reg's loss depends on it -- only the LoRA parameters
        # should have moved.
        assert torch.equal(sae_param_before, sae_param), (
            f"SAE parameter values changed in '{mode}' mode, but this SAE should never be updated here -- "
            "check that make_trainable() wasn't called and that the SAE's parameters were never added to the optimizer."
        )

    print(f"  losses over steps: {losses}")


if __name__ == "__main__":
    for model_key, model_cfg in MODEL_CONFIGS.items():
        print(f"--- {model_key}, frozen SAE (default) ---")
        run_steps(model_key, model_cfg, mode="frozen")
        print(f"--- {model_key}, --joint_sae ---")
        run_steps(model_key, model_cfg, mode="joint_sae")
        print(f"--- {model_key}, --sae_reg ---")
        run_steps(model_key, model_cfg, mode="sae_reg")

    print("Smoke test completed without errors.")
