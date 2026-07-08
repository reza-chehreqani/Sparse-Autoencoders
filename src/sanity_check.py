"""
A minimal smoke test for the Step 2 training pipeline: loads one model, runs a
handful of optimizer steps for the SAE-magnitude-plus-support condition, and
checks that (a) the loss is finite and (b) at least one LoRA parameter actually
received a nonzero gradient -- catching the "forgot to unfreeze the right
parameters" or "SAE encode() accidentally left under no_grad" class of bug
before committing to a full training run.
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

if __name__ == "__main__":
    device = resolve_device(TRAIN_CONFIG["device"])
    for model_key, model_cfg in MODEL_CONFIGS.items():
        print(f"--- {model_key} ---")
        inv_layers = INVARIANCE_LAYERS[model_key][:1]  # just one layer for the smoke test

        saes = load_all_layer_saes(model_cfg, device)
        tokenizer = load_tokenizer(model_cfg)
        peft_model = load_lora_model(model_cfg, TRAIN_CONFIG, device)
        hf_model = peft_model.get_base_model()

        wikitext_texts = load_wikitext_train_text(WIKITEXT_CONFIG["hf_name"], WIKITEXT_CONFIG["hf_config"])
        batcher = WikiTextBatcher(wikitext_texts[:50], tokenizer, WIKITEXT_CONFIG["max_seq_len"], device)

        optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-3)
        lora_param = next(p for p in peft_model.parameters() if p.requires_grad)

        losses, grad_norms = [], []
        for step in range(3):
            lm_batch = batcher.next_batch(2)
            sentences_a = [p[0] for p in SAME_MEANING_PAIRS]
            sentences_b = [p[1] for p in SAME_MEANING_PAIRS]
            acts_a = get_multi_layer_activations(
                hf_model, tokenizer, sentences_a, inv_layers, model_cfg.model_family, model_cfg.hook_side, device
            )
            acts_b = get_multi_layer_activations(
                hf_model, tokenizer, sentences_b, inv_layers, model_cfg.model_family, model_cfg.hook_side, device
            )
            layer = inv_layers[0]
            loss = lm_loss(peft_model, lm_batch) + 1.0 * invariance_loss(
                saes[layer], acts_a[layer], acts_b[layer], space="sae", use_support_term=True
            )
            optimizer.zero_grad()
            loss.backward()
            grad_norm = lora_param.grad.norm().item() if lora_param.grad is not None else 0.0
            optimizer.step()
            losses.append(loss.item())
            grad_norms.append(grad_norm)
            print(f"  step {step}: loss={loss.item():.4f}  a_lora_param_grad_norm={grad_norm:.6f}")

        assert all(l == l for l in losses), "Loss was NaN at some step."  # l == l is False only for NaN
        assert any(g > 0 for g in grad_norms), (
            "No gradient reached the LoRA parameter across any step -- check hook wiring / no_grad usage."
        )
        print(f"  losses over steps: {losses}")

    print("Smoke test completed without errors.")
