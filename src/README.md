# Step 2 (proof of concept) — language-model track

Scope: GPT-2-small, Pythia-70m-deduped, and Gemma-3-270m. This is the first
step in the whole project that actually trains anything (LoRA adapters).

Step 1 is no longer a separate prerequisite: `evaluate.py --run_id baseline`
reproduces everything Step 1 measured (same metrics, same PAWS protocol,
every layer) on the plain pretrained model before any training happens, and
`analyze_and_plot.py`'s depth-profile plots come from that same baseline run.
Layer selection (below) now comes from that baseline eval rather than a
separate Step 1 codebase.

## Before running anything: fill in the real layer selection

`config.py`'s `INVARIANCE_LAYERS` are **placeholders** (`[6]` gpt2-small,
`[3]` pythia-70m-deduped, `[9]` gemma3-270m). Replace them with whichever
layer(s) show the largest delta-AUROC in the baseline eval's depth profile for
each model. If several adjacent layers are close, list all of them — the loss
sums over every layer listed. Results with the placeholders in place don't
mean anything until this is fixed.

## Gemma-3-270m: what's confirmed vs. still open

- **`hf_model_name`**: confirmed as `"google/gemma-3-270m"` (no `-pt` suffix
  — the guessed `-pt` variant 404s).
- **`n_layers`**: confirmed as 18 directly from the shape-check assertion.
- **`d_model`**: still an open guess (640). The assertion checks `n_layers`
  first and stops there if it fails, so it never got to check `d_model` yet —
  next run either passes silently or reports the real value the same way.
- **SAE `release`** (`gemma-scope-2-270m-pt-res-all`) and **`sae_id_template`**
  (`"layer_{layer}_width_16k_l0_small"`): both confirmed working — the
  original `sae_id` guess failed with a `ValueError` that listed every valid
  ID for the release, which is where the correct format came from.

## The five conditions

| Condition | Invariance loss on | Support term | Notes |
|---|---|---|---|
| C0 pretrained baseline | — | — | `evaluate.py --run_id baseline`, no training |
| C1 LM-only fine-tune | — | — | controls for "fine-tuning itself perturbs representations" |
| **C2 raw-activation invariance** | pooled raw activations | n/a | **critical control** — isolates whether SAE-routing adds anything over plain feature-matching |
| C3 SAE invariance, magnitude-only | pooled SAE codes | no | the core idea, without the sparsity-specific piece |
| C4 SAE invariance, magnitude + support | pooled SAE codes | yes (soft-Dice) | tests whether the sparsity-aware term earns its keep over C3 |

Run all five for all three models — a second, stronger test of whether
results generalize across architectures (H5), not just observationally.

## The invariance loss now has two components, not one

Every condition with `use_invariance=True` (C2/C3/C4) now pulls same-meaning
pairs together **and** pushes diff-meaning pairs apart, instead of only the
former:

```
invariance_loss = attractive(same-meaning pairs)       # original term
                 + repulsive(diff-meaning pairs)         # new
                 + support(same-meaning pairs)            # C4 only, unchanged
```

The repulsive term is a margin/hinge loss (`losses.py`'s `invariance_loss`):
zero once a diff-meaning pair is already `repulsive_margin` (default 0.5)
cosine-distance apart, growing only as they get closer than that. This is
what the decision gate previously called "an explicit repulsive term on
diff-meaning pairs" as the fix for representational collapse — it's now
built into every SAE/raw-space run rather than being a fallback to add later.
A margin/hinge is used rather than an unbounded "maximize distance" objective
specifically so the term goes to zero and stops fighting the LM loss once
pairs are adequately separated, rather than pushing forever.

**This changes what C2/C3/C4 compute compared to earlier runs.** If you have
existing `results/step2_llm/` output from before this change, move or rename
that directory before re-running — new runs reuse the same run_id naming and
would land in the same folders, mixing two different loss formulations
together under what looks like one consistent set of results.

## Optional: jointly train the SAE (`--joint_sae`)

By default the SAE stays frozen throughout training (gradients flow *through*
it into the LoRA parameters, but never *into* its own weights — see
`frozen_sae.py`). Passing `--joint_sae` to `train.py` unfreezes the
invariance layer(s)' SAE and adds it to the optimizer (its own, smaller
learning rate: `sae_learning_rate`, default 1e-5), so the dictionary itself
can adapt as the model's activations shift during training — the direct fix
for the frozen-SAE-drift failure mode the training log's
`sae_variance_explained` check was built to catch.

This only does anything for C3/C4 (`train.py` prints a message and ignores
the flag for C1/C2, which don't use the SAE in their loss at all — matched by
`run_ablation.py`'s own `--joint_sae`, which applies the same rule so its
expected run_ids stay in sync with what `train.py` actually saves).

**Why it needs an anchor, and what that anchor is.** A jointly-trained SAE
optimized by the invariance loss alone has two trivial ways to "win" that
have nothing to do with the actual objective:
1. Collapse to an input-independent encoding (perfect invariance, zero
   information content) — nothing in the invariance loss penalizes that.
2. Get denser and denser, since more active latents generally make
   reconstruction (and therefore matching two activations) easier, at the
   cost of defeating the point of using a *sparse* autoencoder at all.

`--joint_sae` therefore always adds two anchor terms from
`FrozenSAE.training_losses(...)`, computed together from a single `encode()`
call on the same per-token activations already fetched for the invariance
loss (no extra forward passes):
- **reconstruction** (MSE, weight `sae_recon_loss_weight`, default 1.0):
  guards against collapse (1) above.
- **sparsity** (L1 on the non-negative code, weight `sparsity_loss_weight`,
  default 1e-4): guards against densification (2) above.

The sparsity weight needs to be small and is only an approximate starting
point — raw L1 magnitude scales with dictionary width (thousands of
dimensions) and is typically far larger than the reconstruction MSE, so an
unweighted or over-weighted term will crush the SAE toward near-zero activity
rather than gently discourage densification. Watch `sae_mean_l0` (average
active latents per token, logged every validation step) after changing this
weight or moving to a wider SAE than the ones used here — it should drift
gradually, not collapse toward 0 or jump to "most of the dictionary counts as
active."

The L1 form itself is a deliberately generic, architecture-agnostic choice.
It's the traditional sparsity loss for "standard"/"gated" SAEs, but the
releases used here are TopK/JumpReLU-style, which enforce sparsity
architecturally (hard top-k selection, or a learned per-feature threshold)
rather than via L1 — so this term acts as an additional soft regularizer for
those rather than a reproduction of their native training objective. If you
know a given release's exact architecture (`sae.cfg.architecture` reports it)
and want to match it exactly, `FrozenSAE.training_losses` is the place to
specialize it.

Trained SAE states are saved to `<run_dir>/trained_sae/layer_<l>.pt` and
picked up automatically by `evaluate.py` when it finds that directory
alongside the adapter — no separate flag needed at evaluation time.

## Data discipline

PAWS `labeled_final`'s own train/validation/test splits are used properly:

- **train**: both same- and diff-meaning pairs are now used *in* the
  invariance loss (attractive and repulsive respectively — see above).
- **validation**: periodic checks during training (every `eval_every` steps):
  LM perplexity, SAE-space same/diff AUROC (the collapse check), mean
  same/diff SAE-cosine-distance, and SAE reconstruction variance explained
  (the frozen-SAE-drift check) — all still plain diagnostics, unaffected by
  the loss formulation changes above.
- **test**: untouched until `evaluate.py`, which re-runs Step 1's *exact*
  measurement pipeline on it — same metrics, same sampling, same protocol —
  so before/after is a clean comparison on data the model never saw.

WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1` — the current canonical,
parquet-backed location) supplies the LM loss, kept deliberately separate
from PAWS so the two loss terms aren't both shaped by the same narrow
sentence pool.

## Implementation notes worth understanding before reading the code

**A differentiable surrogate replaces the discrete support-Jaccard metric for
training.** Step 1's `support_jaccard_distance` thresholds continuous
activations into a boolean mask — a step function with zero gradient almost
everywhere, unusable as a training loss. `losses.py`'s `soft_support_distance`
is a standard continuous relaxation ("soft Dice"), used only for C4's
attractive term. Evaluation (`evaluate.py`) still uses Step 1's exact
discrete metric unchanged, so the *measured outcome* stays comparable, even
though training uses a different (differentiable) proxy along the way.

**Plain PyTorch hooks replace TransformerLens**, across all three
architectures now (`hooked_activations.py`): a forward pre-hook on
`transformer.h[l]` for GPT-2 (`hook_resid_pre`), a forward hook on
`gpt_neox.layers[l]` for Pythia (`hook_resid_post`), and a forward hook on
`model.layers[l]` for Gemma3 (matching Gemma Scope 2's residual-stream site —
also `hook_resid_post`-equivalent). Gemma 3 isn't natively supported by
TransformerLens's standard `HookedTransformer` at all as of this writing, so
this approach is also the more future-proof one for adding further models.
Hooks are registered on `peft_model.get_base_model()` — the actual underlying
model, since LoRA replaces target submodules in place within that same module
tree.

**Nothing in the training path is wrapped in `torch.no_grad()`.** The SAE's
own parameters are frozen via `requires_grad_(False)` by default
(`frozen_sae.py`), not via disabling autograd on `encode()`/`decode()` — the
latter would break gradient flow from the invariance loss back into the LoRA
parameters (and, with `--joint_sae`, into the SAE's own weights). Autograd
still computes gradients *with respect to the SAE's input* regardless of
whether the SAE's own weights require grad — the same pattern as using a
frozen perceptual-loss network. Only diagnostic/eval code (`evaluate.py`, the
SAE-drift check, `reconstruction_variance_explained`) explicitly wraps itself
in `no_grad()`.

**Both `load_lora_model` and `load_model_for_eval` force `torch_dtype=
torch.float32` explicitly**, rather than letting either default from
whatever dtype a checkpoint happens to declare. This was found the hard way:
Pythia-70m-deduped's checkpoint declares float16, which combined with a rare
WikiText raw-markup artifact (a bare `= Header =` line immediately followed
by an EOS token) was enough to occasionally push a tiny 6-layer model's
activations outside float16's representable range and produce
worse-than-random-guessing loss. See `debug_pythia_loss.py` if a new model
ever shows a similar symptom (loss above `ln(vocab_size)`).

## Watch during training: is the (frozen or jointly-trained) SAE still valid?

The SAE starts fit to the *original* model's activations. As training
proceeds, activations drift, and the SAE's reconstruction of them can
degrade — tracked directly via `sae_variance_explained` in the training log.
Without `--joint_sae`, a falling value is a real finding: it means the
frozen-SAE assumption is breaking down for this model/layer/lambda and
results at that point shouldn't be trusted much either way, since the
"ruler" being used to measure the effect is itself no longer working well.
With `--joint_sae`, this same number becomes the direct check on whether the
reconstruction anchor is actually doing its job — it should stay high;
if it doesn't, `sae_recon_loss_weight` likely needs to be larger.

## Running

Needs network access to huggingface.co (models, PAWS, WikiText-2, all three
SAEs) and a GPU for anything beyond the smoke test; written for your own
environment.

```bash
pip install -r requirements.txt

python sanity_check.py              # a few optimizer steps per model, frozen AND --joint_sae paths,
                                       # checks gradients actually flow to the right places in both

python train.py --model gpt2-small --condition C3_sae_magnitude --lam 1.0
python train.py --model gpt2-small --condition C4_sae_magnitude_support --lam 1.0 --joint_sae

python evaluate.py --model gpt2-small --run_id baseline
python evaluate.py --model gpt2-small --run_id gpt2-small__C3_sae_magnitude__lam1.0 \
    --adapter_path results/step2_llm/gpt2-small__C3_sae_magnitude__lam1.0/adapter
# --joint_sae runs: same evaluate.py call, no extra flag -- the trained_sae/
# checkpoint next to the adapter is picked up automatically.

python run_ablation.py --quick                  # one lambda per condition, all three models
python run_ablation.py                            # the full grid (expensive)
python run_ablation.py --joint_sae --quick          # same, with joint SAE training for C3/C4
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
- **Collapse persists even with the repulsive term** (`mean_diff_sae_cos`
  still drops toward `mean_same_sae_cos` in the training log) → try a larger
  `repulsive_margin` before concluding the approach doesn't work; the current
  default (0.5) is a starting point, not a tuned value.
- **`sae_variance_explained` degrades badly without `--joint_sae`, but stays
  healthy with it** → the frozen-SAE assumption was the actual bottleneck for
  that model/layer, and joint training is worth keeping on by default there.
- **`sae_mean_l0` collapses toward 0 or balloons toward the full dictionary
  width with `--joint_sae`** → `sparsity_loss_weight` is off (too high or too
  low respectively) for this SAE's actual scale; retune before trusting that
  run's other numbers.
