# Step 2 (proof of concept) — language-model track

Scope: GPT-2-small and Pythia-70m-deduped, continuing directly from Step 1.
This is the first step in the whole project that actually trains anything
(LoRA adapters); everything before it only measured frozen models.

## Before running anything: fill in the real layer selection

`config.py`'s `INVARIANCE_LAYERS` are **placeholders** (`[6]` for gpt2-small,
`[3]` for pythia-70m-deduped). Replace them with whichever layer(s) showed the
largest delta-AUROC in Step 1's `depth_profile.png` for each model. If two or
three adjacent layers are close, list all of them — the loss sums over every
layer listed. Running with the placeholders in place will produce results, but
they won't mean anything until this is fixed.

## The five conditions

| Condition | Invariance loss on | Support term | Notes |
|---|---|---|---|
| C0 pretrained baseline | — | — | `evaluate.py --run_id baseline`, no training |
| C1 LM-only fine-tune | — | — | controls for "fine-tuning itself perturbs representations" |
| **C2 raw-activation invariance** | pooled raw activations | n/a | **critical control** — isolates whether SAE-routing adds anything over plain feature-matching |
| C3 SAE invariance, magnitude-only | pooled SAE codes | no | the core idea, without the sparsity-specific piece |
| C4 SAE invariance, magnitude + support | pooled SAE codes | yes (soft-Dice) | tests whether the sparsity-aware term earns its keep over C3 |

Run all five for both models — a second, stronger test of whether results
generalize across architectures (H5), not just observationally as in Step 1.

## Data discipline

PAWS `labeled_final`'s own train/validation/test splits are used properly this
time, since actual optimization is involved:

- **train**: only same-meaning pairs are used *in* the invariance loss — it
  only pulls same-meaning pairs together, it never pushes diff-meaning pairs
  apart. Diff-meaning training-split pairs are loaded but never optimized
  against.
- **validation**: periodic checks during training (every `eval_every` steps):
  LM perplexity, SAE-space same/diff AUROC (the collapse check), and SAE
  reconstruction variance explained (the frozen-SAE-drift check).
- **test**: untouched until `evaluate.py`, which re-runs Step 1's *exact*
  measurement pipeline on it — same metrics, same sampling, same protocol —
  so before/after is a clean comparison on data the model never saw.

WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1` — the current canonical,
parquet-backed location; the unprefixed `wikitext` repo has had loading-script
issues since HF deprecated script-based dataset loading) supplies the LM loss,
kept deliberately separate from PAWS so the two loss terms aren't both shaped
by the same narrow sentence pool.

## Three implementation decisions worth understanding before reading the code

**A differentiable surrogate replaces the discrete support-Jaccard metric for
training.** Step 1's `support_jaccard_distance` thresholds continuous
activations into a boolean mask — a step function with zero gradient almost
everywhere, unusable as a training loss. `losses.py`'s `soft_support_distance`
is a standard continuous relaxation ("soft Dice"), used only for C4's training
signal. Evaluation (`evaluate.py`) still uses Step 1's exact discrete metric
unchanged, so the *measured outcome* stays comparable to Step 1, even though
training uses a different (differentiable) proxy along the way.

**Plain PyTorch hooks replace TransformerLens.** `peft`'s LoRA is built around
plain `transformers` module structure; mixing in TransformerLens here would
fight the tooling. `hooked_activations.py` re-implements the same two hook
points Step 1 established (`hook_resid_pre` via a forward pre-hook on
`transformer.h[l]` for GPT-2, `hook_resid_post` via a forward hook on
`gpt_neox.layers[l]` for Pythia) directly on the HF module tree, and hooks are
registered on `peft_model.get_base_model()` — the actual underlying model,
since LoRA replaces target submodules in place within that same module tree.

**Nothing in the training path is wrapped in `torch.no_grad()`.** The SAE's own
parameters are frozen via `requires_grad_(False)` (`frozen_sae.py`), not via
disabling autograd on its `encode()`/`decode()` calls — the latter would break
gradient flow from the invariance loss back into the LoRA parameters. Autograd
still computes gradients *with respect to the SAE's input* even though the
SAE's own weights never accumulate a gradient — the same pattern as using a
frozen perceptual-loss network. Only diagnostic/eval code (`evaluate.py`, the
SAE-drift check in `frozen_sae.py`) explicitly wraps itself in `no_grad()`.

## Watch during training: is the frozen SAE still valid?

The SAE was fit to the *original* model's activations. As LoRA adapts the
model, activations drift, and the frozen SAE's reconstruction of them can
degrade — tracked directly via `sae_variance_explained` in the training log. A
small-rank (`lora_r=8`), short (`max_steps=500`) run should keep this small; if
it drops noticeably, that's a real finding, not noise — it means the
frozen-SAE assumption needs revisiting (shorter runs, smaller rank, or jointly
training / periodically refreshing the SAE) before scaling up.

## Running

Needs network access to huggingface.co (models, PAWS, WikiText-2, both SAEs)
and a GPU for anything beyond the smoke test; written for your own environment.

```bash
pip install -r requirements.txt

python sanity_check.py              # a few optimizer steps, checks gradients actually flow
python train.py --model gpt2-small --condition C3_sae_magnitude --lam 1.0
python evaluate.py --model gpt2-small --run_id baseline
python evaluate.py --model gpt2-small --run_id gpt2-small__C3_sae_magnitude__lam1.0 \
    --adapter_path results/step2_llm/gpt2-small__C3_sae_magnitude__lam1.0/adapter

python run_ablation.py --quick      # one lambda per condition, both models -- do this before...
python run_ablation.py               # ...the full grid (~27 training runs, expensive)
```

## Decision gate

- **C3/C4 widen the test-set same-vs-diff AUROC gap more than C2, at
  comparable perplexity** → SAE-routing earns its keep during actual training,
  not just observationally. Proceed to scale up.
- **C4 clearly beats C3** → the support term is pulling weight, not just
  magnitude.
- **C2 ≈ C3/C4** → SAE space doesn't help once you're actually optimizing
  against it. Worth reporting honestly, and worth redirecting the project's
  framing toward the interpretability angle (which concepts changed) rather
  than a training/robustness claim.
- **Every condition shows collapse** (`mean_diff_sae_cos` drops about as much
  as `mean_same_sae_cos` in the training log) → the loss formulation itself
  needs an explicit repulsive term on diff-meaning pairs before anything else
  is worth measuring.
