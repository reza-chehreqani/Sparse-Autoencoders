"""
Trains one (model, condition, lambda) combination for Step 2.

Each step draws one WikiText batch and (for C2-C4) one PAWS same-meaning batch
AND one PAWS diff-meaning batch, combining all of it into a single backward pass:

    total_loss = lm_loss(wikitext_batch)
               + lambda * [ attractive(same_batch) + repulsive(diff_batch) (+ support) ]
               + (only with --joint_sae) sae_recon_loss_weight * reconstruction
               + (only with --joint_sae) sparsity_loss_weight * sparsity
               + (only with --sae_reg)   sae_recon_loss_weight * reconstruction

Three mutually exclusive ways to handle the SAE during training (default is
the first):
  1. Frozen, no extra loss term. Simplest, but sae_variance_explained in the
     training log can degrade as the model's activations drift away from what
     the SAE was originally fit to -- watch for this rather than assume it away.
  2. --joint_sae: let the SAE adapt to the drifting model. Keeps reconstruction
     high by construction, but the SAE's own dictionary changes, so the
     baseline and trained checkpoints are evaluated with two different SAEs
     (see trained_sae/ below) -- and some of any measured invariance gain
     could come from the SAE's own projection adapting rather than the
     model's actual representations changing.
  3. --sae_reg: keep the SAE frozen (nothing to save, nothing changes) and
     instead add its reconstruction loss on the model's current activations
     as a regularizer on the MODEL -- gradient flows through the frozen SAE
     into the LoRA parameters, penalizing drift away from the region the SAE
     can reconstruct, the same way a KL-to-reference-policy penalty works in
     RL fine-tuning. The SAE used for evaluation is then guaranteed identical,
     unchanged, before and after -- the more direct fix if the goal is
     specifically keeping the SAE valid as a fixed measurement yardstick.

See losses.py for why the repulsive term uses a margin/hinge rather than an
unbounded "maximize distance" objective, and frozen_sae.py's training_losses
for the reconstruction/sparsity formulas used by both --joint_sae and
--sae_reg (sae_reg only uses the reconstruction half -- there's no SAE
dictionary being trained for a sparsity term to regularize).

Produces, under results/step2_llm/<model>__<condition>__lam<lambda>[__jointsae|__saereg]/:
  - adapter/               the trained LoRA adapter (not the full model)
  - trained_sae/            (only with --joint_sae) one state dict per invariance
                            layer's SAE, as it ended training -- evaluate.py picks
                            these up automatically if present for a given run_id.
                            --sae_reg has no equivalent: its SAE never changes.
  - training_log.json      periodic validation checks: LM perplexity, SAE-space
                            same/diff AUROC (the collapse check -- this should
                            WIDEN over training, not shrink), mean same/diff
                            SAE-cosine-distance, SAE reconstruction variance
                            explained (the frozen-SAE-drift check -- this is
                            the number --sae_reg and --joint_sae are each
                            trying to keep healthy, by two different means),
                            average active latents per token (sae_mean_l0 --
                            the sparsity check, only meaningful with
                            --joint_sae), and the loss components.

Usage:
    python train.py --model gpt2-small --condition C3_sae_magnitude --lam 1.0
    python train.py --model gpt2-small --condition C4_sae_magnitude_support --lam 1.0 --joint_sae
    python train.py --model gpt2-small --condition C4_sae_magnitude_support --lam 1.0 --sae_reg
"""

import argparse
import json
import os
import random

import torch

from config import CONDITION_SPECS, INVARIANCE_LAYERS, MODEL_CONFIGS, PAWS_CONFIG, TRAIN_CONFIG, WIKITEXT_CONFIG
from frozen_sae import load_layer_saes
from hf_model_loading import load_lora_model, load_tokenizer, resolve_device
from hooked_activations import capture_layer_activations_during, get_multi_layer_activations
from lm_dataset import WikiTextBatcher, load_wikitext_train_text
from losses import invariance_loss, lm_loss, pool_sae_codes, ScaledCosineBCELoss
from metrics import cosine_distance, discrimination_auroc
from pairs_dataset import load_paws_split


def _grad_norm(loss: torch.Tensor, params: list) -> float:
    """L2 norm of the gradient `loss` would contribute to `params`, computed
    without disturbing the real optimizer step below: retain_graph=True keeps
    the graph alive for that later backward() call, and autograd.grad returns
    gradients directly rather than accumulating into params[i].grad.

    This is the actual diagnostic for "which loss term is driving training" --
    loss *magnitude* is not a reliable proxy for gradient magnitude, since
    different terms can have very different gradient scaling at the same
    nominal loss value. This is precisely the diagnostic that would have
    caught the raw-MSE-reconstruction-dominating-total_loss bug earlier: the
    first full ablation's training_log.json had all the loss *values* right
    there, but nothing showing what fraction of the actual parameter update
    each one was responsible for.

    Only call this periodically (see call site) -- each call is an extra
    backward-shaped pass through the graph, not free.
    """
    if not isinstance(loss, torch.Tensor) or loss.grad_fn is None:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq_sum = sum(g.pow(2).sum() for g in grads if g is not None)
    return float(sq_sum.sqrt().item()) if isinstance(sq_sum, torch.Tensor) else 0.0


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

    # 1. Capture WikiText activations for ALL invariance layers during perplexity pass
    lm_loss_value, wikitext_acts = capture_layer_activations_during(
        hf_model,
        inv_layers,                     # <-- changed from [layer] to inv_layers
        model_family,
        hook_side,
        lambda: lm_loss(peft_model, val_batch),
    )
    ppl = torch.exp(lm_loss_value).item()

    sentences = dict(
        a_same=[p.sentence_a for p in val_same_pairs],
        b_same=[p.sentence_b for p in val_same_pairs],
        a_diff=[p.sentence_a for p in val_diff_pairs],
        b_diff=[p.sentence_b for p in val_diff_pairs],
    )
    # one forward pass per sentence group, all invariance layers captured at once
    acts_by_layer = {
        k: get_multi_layer_activations(hf_model, tokenizer, v, inv_layers, model_family, hook_side, device)
        for k, v in sentences.items()
    }

    per_layer = {}
    for layer in inv_layers:
        sae = saes[layer]
        a_same = acts_by_layer["a_same"][layer]
        b_same = acts_by_layer["b_same"][layer]
        a_diff = acts_by_layer["a_diff"][layer]
        b_diff = acts_by_layer["b_diff"][layer]

        same_cos = cosine_distance(pool_sae_codes(sae, a_same), pool_sae_codes(sae, b_same)).cpu().numpy()
        diff_cos = cosine_distance(pool_sae_codes(sae, a_diff), pool_sae_codes(sae, b_diff)).cpu().numpy()
        auroc_sae, _ = discrimination_auroc(same_cos, diff_cos)

        paws_tokens = torch.cat(a_same + b_same + a_diff + b_diff, dim=0)
        wikitext_tokens_flat = wikitext_acts[layer].reshape(-1, wikitext_acts[layer].shape[-1])
        drift_check_tokens = torch.cat([paws_tokens, wikitext_tokens_flat], dim=0)

        var_explained = sae.reconstruction_variance_explained(drift_check_tokens)
        mean_l0 = sae.mean_l0(drift_check_tokens)

        per_layer[layer] = dict(
            auroc_sae=float(auroc_sae),
            same_cos=same_cos,
            diff_cos=diff_cos,
            var_explained=float(var_explained),
            mean_l0=float(mean_l0),
        )

    aurocs = [v["auroc_sae"] for v in per_layer.values()]
    same_cos_means = [v["same_cos"].mean() for v in per_layer.values()]
    diff_cos_means = [v["diff_cos"].mean() for v in per_layer.values()]
    var_explained_vals = [v["var_explained"] for v in per_layer.values()]
    mean_l0_vals = [v["mean_l0"] for v in per_layer.values()]

    peft_model.train()
    return dict(
        perplexity=ppl,
        auroc_sae=float(sum(aurocs) / len(aurocs)),
        mean_same_sae_cos=float(sum(same_cos_means) / len(same_cos_means)),
        mean_diff_sae_cos=float(sum(diff_cos_means) / len(diff_cos_means)),
        sae_variance_explained=float(sum(var_explained_vals) / len(var_explained_vals)),
        sae_mean_l0=float(sum(mean_l0_vals) / len(mean_l0_vals)),
    )


def run(model_key: str, condition: str, lam: float, joint_sae: bool, sae_reg: bool, seed: int = 0) -> None:
    torch.manual_seed(seed)
    assert not (joint_sae and sae_reg), (
        "--joint_sae and --sae_reg are mutually exclusive: they're two different answers to the same "
        "problem (joint_sae adapts the SAE to the drifting model; sae_reg holds the SAE fixed and "
        "penalizes the model for drifting away from it). Pick one."
    )
    model_cfg = MODEL_CONFIGS[model_key]
    spec = CONDITION_SPECS[condition]
    inv_layers = INVARIANCE_LAYERS[model_key]
    device = resolve_device(TRAIN_CONFIG["device"])
    mode_tag = "/joint_sae" if joint_sae else "/sae_reg" if sae_reg else ""
    tag = f"[{model_key}/{condition}/lam={lam}{mode_tag}]"

    # sae_option_applies = spec["use_invariance"] and spec["space"] == "sae"
    # if joint_sae and not sae_option_applies:
    #     print(f"{tag} --joint_sae has no effect on {condition}: its invariance loss doesn't run through the "
    #           f"SAE. Proceeding without it.")
    #     joint_sae = False
    # if sae_reg and not sae_option_applies:
    #     print(f"{tag} --sae_reg has no effect on {condition}: its invariance loss doesn't run through the "
    #           f"SAE, so there's nothing to regularize the model against. Proceeding without it.")
    #     sae_reg = False

    print(f"{tag} loading SAEs ...")
    saes = load_layer_saes(model_cfg, device, inv_layers)
    if joint_sae:
        for l in inv_layers:
            saes[l].make_trainable()
    # sae_reg: SAEs stay frozen (the default) -- its reconstruction loss is used
    # as a regularizer on the MODEL, computed through the SAE without ever
    # calling make_trainable() on it, same frozen-module-inside-a-loss pattern
    # already used for the invariance loss itself.

    print(f"{tag} loading LoRA model ...")
    tokenizer = load_tokenizer(model_cfg)
    peft_model = load_lora_model(model_cfg, TRAIN_CONFIG, device)
    hf_model = peft_model.get_base_model()

    print(f"{tag} loading data ...")
    train_pairs = load_paws_split(**_paws_split_kwargs("train"))
    train_same = [p for p in train_pairs if p.same_meaning]
    train_diff = [p for p in train_pairs if not p.same_meaning]
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

    if spec["use_bce_term"]:
    # Initialize the BCE Criterion
        bce_criterion = ScaledCosineBCELoss().to(device)

        # Append BCE Criterion parameters to the main optimizer group
        param_groups = [dict(
            params=[p for p in peft_model.parameters() if p.requires_grad] + list(bce_criterion.parameters()), 
            lr=TRAIN_CONFIG["learning_rate"]
        )]
    else:
        param_groups = [dict(params=[p for p in peft_model.parameters() if p.requires_grad], lr=TRAIN_CONFIG["learning_rate"])]
    
    if joint_sae:
        sae_params = [p for l in inv_layers for p in saes[l].parameters()]
        param_groups.append(dict(params=sae_params, lr=TRAIN_CONFIG["sae_learning_rate"]))
    optimizer = torch.optim.AdamW(param_groups)

    # For the periodic grad-norm diagnostic below -- LoRA params only (not the
    # SAE's own params, even under --joint_sae): the question this answers is
    # "how hard is each loss term pulling on the MODEL", which is what's
    # actually at stake for representation drift/collapse.
    trainable_model_params = [p for p in peft_model.parameters() if p.requires_grad]

    rng_state: dict = {}
    log = []
    for step in range(TRAIN_CONFIG["max_steps"]):
        lm_batch = wikitext_train_batcher.next_batch(TRAIN_CONFIG["batch_size_lm"])
        if joint_sae or sae_reg:
            # Piggybacks on the LM loss's own forward pass to also capture the
            # invariance layer(s)' activations on this WikiText batch, at no
            # extra forward-pass cost -- see hooked_activations.
            # capture_layer_activations_during's docstring for why this
            # matters: without it, the SAE anchor/regularizer only ever sees
            # the narrow PAWS sentence distribution, never the general text
            # the LM loss is actually training on every step.
            loss_lm, lm_layer_acts = capture_layer_activations_during(
                hf_model, inv_layers, model_cfg.model_family, model_cfg.hook_side,
                lambda: lm_loss(peft_model, lm_batch),
            )
            recon_loss = torch.tensor(0.0, device=device)
            sparsity_loss = torch.tensor(0.0, device=device)
            for l in inv_layers:
                lm_acts_flat = lm_layer_acts[l].reshape(-1, lm_layer_acts[l].shape[-1])
                sae_losses = saes[l].training_losses(lm_acts_flat)
                recon_loss = recon_loss + sae_losses["reconstruction"]
                if joint_sae:
                    sparsity_loss = sparsity_loss + sae_losses["sparsity"]
            loss_lm_sae = TRAIN_CONFIG["sae_recon_loss_weight"] * recon_loss + TRAIN_CONFIG["sparsity_loss_weight"] * sparsity_loss

        else:
            loss_lm = lm_loss(peft_model, lm_batch)
            loss_lm_sae = torch.tensor(0.0)

        if spec["use_invariance"]:
            same_batch = sample_batch(train_same, TRAIN_CONFIG["batch_size_invariance"], rng_state, "train_same")
            diff_batch = sample_batch(train_diff, TRAIN_CONFIG["batch_size_invariance"], rng_state, "train_diff")

            same_acts_a = get_multi_layer_activations(
                hf_model, tokenizer, [p.sentence_a for p in same_batch], inv_layers,
                model_cfg.model_family, model_cfg.hook_side, device,
            )
            same_acts_b = get_multi_layer_activations(
                hf_model, tokenizer, [p.sentence_b for p in same_batch], inv_layers,
                model_cfg.model_family, model_cfg.hook_side, device,
            )
            diff_acts_a = get_multi_layer_activations(
                hf_model, tokenizer, [p.sentence_a for p in diff_batch], inv_layers,
                model_cfg.model_family, model_cfg.hook_side, device,
            )
            diff_acts_b = get_multi_layer_activations(
                hf_model, tokenizer, [p.sentence_b for p in diff_batch], inv_layers,
                model_cfg.model_family, model_cfg.hook_side, device,
            )

            loss_inv = torch.tensor(0.0, device=device)
            inv_components: dict = {}
            for l in inv_layers:
                l_total, l_components = invariance_loss(
                    saes[l], same_acts_a[l], same_acts_b[l], diff_acts_a[l], diff_acts_b[l], device,
                    spec["space"], spec["use_support_term"], spec["use_rank_term"], spec["use_bce_term"],
                    bce_criterion if spec["use_bce_term"] else None
                )
                loss_inv = loss_inv + l_total
                for k, v in l_components.items():
                    inv_components[f"layer{l}_{k}"] = v


            # if joint_sae or sae_reg:
            #     # Shared across both modes: PAWS (same + diff meaning) activations
            #     # plus, when available, this step's WikiText activations -- the
            #     # anchor/regularizer sees the same mix of distributions the rest
            #     # of training does, not just the narrower PAWS slice.
            #     anchor_acts = {}
            #     for l in inv_layers:
            #         paws_acts = torch.cat(same_acts_a[l] + same_acts_b[l] + diff_acts_a[l] + diff_acts_b[l], dim=0)
            #         if lm_layer_acts is not None:
            #             lm_acts_flat = lm_layer_acts[l].reshape(-1, lm_layer_acts[l].shape[-1])
            #             anchor_acts[l] = torch.cat([paws_acts, lm_acts_flat], dim=0)
            #         else:
            #             anchor_acts[l] = paws_acts

            if joint_sae:
                recon_loss = torch.tensor(0.0, device=device)
                sparsity_loss = torch.tensor(0.0, device=device)
                for l in inv_layers:
                    all_acts = torch.cat(same_acts_a[l] + same_acts_b[l] + diff_acts_a[l] + diff_acts_b[l], dim=0)
                    sae_losses = saes[l].training_losses(all_acts)
                    recon_loss = recon_loss + sae_losses["reconstruction"]
                    sparsity_loss = sparsity_loss + sae_losses["sparsity"]
                loss_inv_sae = TRAIN_CONFIG["sae_recon_loss_weight"] * recon_loss + TRAIN_CONFIG["sparsity_loss_weight"] * sparsity_loss
                inv_components["sae_recon_loss"] = recon_loss.item()
                inv_components["sae_sparsity_loss"] = sparsity_loss.item()
            else:
                loss_inv_sae = torch.tensor(0.0)

            if sae_reg:
                # Same reconstruction term as joint_sae's anchor, but the SAE
                # was never made trainable, so this gradient has nowhere to go
                # except back through the (frozen) SAE into the activations --
                # i.e. into the LoRA parameters that produced them (both the
                # PAWS-batch and, via lm_layer_acts, the WikiText-batch ones).
                # This is what actually regularizes the MODEL to stay in the
                # region the fixed SAE can reconstruct, rather than adapting
                # the SAE to wherever the model drifts. No sparsity term here:
                # nothing about the SAE's own dictionary is being trained, so
                # there's no densification failure mode to guard against --
                # only reconstruction fidelity is the thing at risk.
                recon_loss = torch.tensor(0.0, device=device)
                for l in inv_layers:
                    all_acts = torch.cat(same_acts_a[l] + same_acts_b[l] + diff_acts_a[l] + diff_acts_b[l], dim=0)
                    recon_loss = recon_loss + saes[l].training_losses(all_acts)["reconstruction"]
                loss_inv_sae = TRAIN_CONFIG["sae_recon_loss_weight"] * recon_loss
                inv_components["sae_reg_recon_loss"] = recon_loss.item()
            else:
                loss_inv_sae = torch.tensor(0.0)
        else:
            loss_inv = torch.tensor(0.0)
            inv_components = {}
            loss_inv_sae = torch.tensor(0.0)
            
        total_loss = loss_lm + loss_lm_sae + lam * loss_inv + loss_inv_sae

        do_diag = step % TRAIN_CONFIG["eval_every"] == 0 or step == TRAIN_CONFIG["max_steps"] - 1
        grad_diag = {}
        if do_diag:
            # Computed BEFORE zero_grad()/backward() below: each _grad_norm
            # call uses retain_graph=True precisely so it doesn't consume the
            # graph the real backward() still needs, and doesn't touch
            # .grad on any param (autograd.grad returns gradients directly).
            grad_diag = dict(
                gradnorm_lm=_grad_norm(loss_lm, trainable_model_params),
                gradnorm_lm_sae=_grad_norm(loss_lm_sae, trainable_model_params),
                gradnorm_inv=_grad_norm(lam * loss_inv, trainable_model_params) if spec["use_invariance"] else 0.0,
                gradnorm_inv_sae=_grad_norm(lam * loss_inv_sae, trainable_model_params) if spec["use_invariance"] else 0.0,
            )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if do_diag:
            checks = run_validation_checks(
                peft_model, hf_model, tokenizer, saes, inv_layers, model_cfg.model_family, model_cfg.hook_side,
                val_same, val_diff, wikitext_val_batcher, device,
            )
            checks.update(step=step, loss_lm=float(loss_lm.item()), loss_lm_sae=float(loss_lm_sae.item()), loss_inv=float(loss_inv.item()), loss_inv_sae=float(loss_inv_sae.item()), total_loss=float(total_loss.item()), **inv_components, **grad_diag)
            log.append(checks)
            print(
                f"  step {step}: lm_loss={checks['loss_lm']:.3f} inv_loss={checks['loss_inv']:.4f} "
                f"ppl={checks['perplexity']:.2f} auroc_sae={checks['auroc_sae']:.3f} "
                f"same_cos={checks['mean_same_sae_cos']:.3f} diff_cos={checks['mean_diff_sae_cos']:.3f} "
                f"sae_var_explained={checks['sae_variance_explained']:.3f} sae_mean_l0={checks['sae_mean_l0']:.1f}"
            )
            print(
                f"    grad norms: lm={grad_diag['gradnorm_lm']:.4f}  lm_sae={grad_diag['gradnorm_lm_sae']:.4f}  "
                f"inv={grad_diag['gradnorm_inv']:.4f}  inv_sae={grad_diag['gradnorm_inv_sae']:.4f}"
                + ("  <-- inv_sae dominating inv, anchor still too strong" if grad_diag['gradnorm_inv_sae'] > 5 * max(grad_diag['gradnorm_inv'], 1e-8) else "")
            )

    run_id = f"{model_key}__{condition}__lam{lam}" #+ ("__jointsae" if joint_sae else "__saereg" if sae_reg else "")
    out_dir = os.path.join(TRAIN_CONFIG["output_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(os.path.join(out_dir, "adapter"))
    if joint_sae:
        # sae_reg has no equivalent save step -- its SAE never changes, so
        # evaluate.py's fresh pretrained SAE is already the correct one to use.
        sae_dir = os.path.join(out_dir, "trained_sae")
        os.makedirs(sae_dir, exist_ok=True)
        for l in inv_layers:
            saes[l].save(os.path.join(sae_dir, f"layer_{l}.pt"))
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[{run_id}] done, saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--condition", required=True, choices=list(CONDITION_SPECS.keys()))
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--joint_sae", action="store_true",
                         help="Also train the invariance layer(s)' SAE, instead of keeping it frozen. "
                              "Only meaningful for conditions whose invariance loss runs through the SAE (C3/C4). "
                              "Mutually exclusive with --sae_reg.")
    parser.add_argument("--sae_reg", action="store_true",
                         help="Keep the SAE frozen (as by default) but add its reconstruction loss on the "
                              "model's current activations to the total loss, regularizing the MODEL to stay "
                              "in the region the fixed SAE can reconstruct, rather than letting the SAE go "
                              "stale as the model drifts. Only meaningful for C3/C4. Mutually exclusive with "
                              "--joint_sae. This is the recommended default if the goal is specifically to "
                              "keep the SAE valid for evaluation, since the SAE used at evaluation time is "
                              "then guaranteed identical to the one used at the start of training.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.model, args.condition, args.lam, args.joint_sae, args.sae_reg, args.seed)
