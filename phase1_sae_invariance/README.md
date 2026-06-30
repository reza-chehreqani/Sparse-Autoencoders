# Phase 1: SAE Paraphrase Invariance Across Layers

This is Phase 1 of the *Layer-Stratified SAE Feature Invariance for
Semantic Robustness* study. It answers one question with a controlled
experiment: **does SAE-feature similarity between paraphrase pairs
increase with layer depth, more than raw-activation similarity does?**

If the answer is yes (and the literature points that way, but no one
has run this as a primary, controlled, all-layer experiment), it
justifies Phase 2's layer-stratified invariance training objective:
constrain deep layers strongly, leave shallow layers alone.

## Why PAWS

PAWS pairs sentences with **high lexical overlap but different
meaning** as negatives, and true paraphrases as positives. This is
the control that makes the result interpretable. Without it, "high
similarity" could just mean "shares a lot of the same words" rather
than "encodes the same meaning." See the docstring in `data.py` for
more detail.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # or: export HF_TOKEN=...
```

You must also accept the Gemma license on Hugging Face
(huggingface.co/google/gemma-2-2b) with the account you log in with --
gated repo, one-click accept.

**Hardware**: a single GPU with >= 16 GB VRAM is recommended (e.g. a
T4/L4 on Colab, or any 16GB+ consumer/datacenter card). The pipeline
loads one Gemma Scope SAE at a time (see `extract.py`), not all 26 at
once, specifically to keep this feasible on a single GPU. CPU-only
will work but will be slow.

**Download size**: gemma-2-2b (~5GB) plus 26 canonical 16k-width SAEs
(roughly 150-300MB each, so a few GB total on first run; cached by
`huggingface_hub` afterward).

## Run order

```bash
# 1. Sanity-check the analysis/plotting code with synthetic data.
#    No GPU, no downloads, no HF login needed for this step.
python test_smoke.py

# 2. Confirm the SAE id naming matches your installed sae_lens version.
python list_available_saes.py
# -> compare the printed sae_id pattern against config.py's template;
#    edit config.py if they differ.

# 3. Run the real pipeline.
python run_phase1.py
```

Step 3, with the default config (250 paraphrase + 250 non-paraphrase
pairs, 26 layers), should take somewhere in the range of 10-30 minutes
on a single mid-range GPU, dominated by the one-time SAE downloads and
Stage A's 1000 forward passes. Stage B (the SAE encoding itself) is
cheap -- each SAE only has to encode ~1000 single vectors.

If you hit GPU memory issues, reduce `n_pairs_per_class` in
`config.py` first; the SAE loading strategy already minimizes memory,
so the dataset size is the next lever.

## Outputs

All written to `./phase1_outputs/`:

| File | Contents |
|---|---|
| `similarity_table.csv` | Every pair x every layer: raw and SAE cosine similarity |
| `layer_summary.csv` | Mean +/- SEM per layer, per group -- **this feeds Phase 2** |
| `separation_gap.csv` | The headline metric: paraphrase-vs-non-paraphrase gap per layer |
| `stats.json` | Spearman correlation (gap vs. depth) and Mann-Whitney test at the deepest layer |
| `similarity_by_layer.png` | Two-panel plot: raw activations vs. SAE features, both groups |
| `separation_gap.png` | The single plot that makes or breaks the headline claim |

## How to read the result

Open `separation_gap.png`. The hypothesis is confirmed if:

1. The **green (SAE-feature) curve rises** with layer depth.
2. It **pulls ahead of the gray (raw-activation) curve**, especially
   in the back half of the model.
3. `stats.json` shows `sae_gap_vs_depth_spearman_p < 0.05` with a
   positive `rho`, and ideally a much weaker/non-significant
   `raw_gap_vs_depth_spearman_p`.
4. `deepest_layer_mannwhitney_p < 0.05` -- paraphrase and
   non-paraphrase pairs are significantly separated by SAE features at
   the final layer, despite both groups sharing high lexical overlap
   (the PAWS negatives).

If the SAE-feature curve does *not* clearly separate from the raw
curve, that is also a real, useful, reportable result -- it would mean
the semantic-stratigraphy hypothesis needs revision before Phase 2 is
built on top of it. Either outcome is worth writing up; check the
result honestly before moving on.

## What feeds into Phase 2

`layer_summary.csv`'s `sae_mean` column (for `is_paraphrase=True`),
or more precisely the *shape* of `separation_gap.csv`'s `sae_gap`
column, is what Phase 2 will normalize into the per-layer loss weight
lambda(L): layers with a small or flat gap get lambda ~ 0 (don't
constrain them -- they're surface-level), layers with a large gap get
high lambda (constrain them strongly -- they're where meaning lives).

## Files in this project

```
config.py                  All tunable parameters
data.py                    PAWS pair loading
extract.py                 Two-stage model + SAE feature extraction
analyze.py                 Per-layer statistics
visualize.py                Plotting
list_available_saes.py     Verifies sae_lens naming before a real run
test_smoke.py               Synthetic-data test of analyze.py/visualize.py (run first, no GPU needed)
run_phase1.py               Main entry point
```
