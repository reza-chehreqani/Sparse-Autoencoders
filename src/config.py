"""
Configuration for Step 2 (proof of concept), language-model track.

Model/SAE facts here (release names, hook points, layer counts) match Step 1's
config.py -- duplicated rather than imported cross-directory so Step 2 can run
independently of Step 1's folder layout (see README). Everything else here is
new: LoRA target modules, which layer(s) to train the invariance loss against,
the five-condition ablation grid, and the two datasets involved.
"""

from dataclasses import dataclass


@dataclass
class SAEResource:
    release: str


@dataclass
class ModelConfig:
    name: str
    hf_model_name: str      # plain HuggingFace model id
    model_family: str        # "gpt2", "gptneox", or "gemma3" -- selects hook-registration logic
    n_layers: int
    d_model: int
    sae: SAEResource
    sae_id_template: str      # SAELens sae_id, e.g. "blocks.{layer}.hook_resid_pre"
    hook_side: str              # "pre" (hook_resid_pre: hook on block INPUT) or
    # "post" (hook_resid_post: hook on block OUTPUT) -- must match what the SAE
    # release above was actually trained on, not chosen freely.
    lora_target_modules: list


MODEL_CONFIGS = {
    "gpt2-small": ModelConfig(
        name="gpt2-small",
        hf_model_name="gpt2",
        model_family="gpt2",
        n_layers=12,
        d_model=768,
        sae=SAEResource(release="gpt2-small-res-jb"),
        sae_id_template="blocks.{layer}.hook_resid_pre",
        hook_side="pre",
        # NOTE: "c_proj" is the exact submodule name of BOTH the attention output
        # projection and the MLP output projection in HF's GPT-2 implementation --
        # peft matches by name suffix, so this also LoRA-wraps the MLP projection,
        # not attention alone. That's standard practice for LoRA-on-GPT-2, not a
        # bug, but worth knowing rather than discovering by surprise.
        lora_target_modules=["c_attn", "c_proj"],
    ),
    "pythia-70m-deduped": ModelConfig(
        name="pythia-70m-deduped",
        hf_model_name="EleutherAI/pythia-70m-deduped",
        model_family="gptneox",
        n_layers=6,
        d_model=512,
        sae=SAEResource(release="pythia-70m-deduped-res-sm"),
        sae_id_template="blocks.{layer}.hook_resid_post",
        hook_side="post",
        lora_target_modules=["query_key_value", "dense"],
    ),
    "gemma3-270m": ModelConfig(
        name="gemma3-270m",
        # Best guess following the google/gemma-3-{size}-pt naming convention
        # used for the other Gemma 3 sizes' base checkpoints; verify with
        # sanity_check.py -- a wrong repo id fails immediately and clearly
        # (repository not found), it can't fail silently.
        hf_model_name="google/gemma-3-270m",
        model_family="gemma3",
        # n_layers/d_model: sources disagree (one third-party writeup says 12
        # layers / hidden_size 1024; a HF discussion referencing the model's
        # own config.json suggests hidden_size may be 640). Rather than pick
        # one and hope, _check_model_shape in hf_model_loading.py asserts both
        # of these against the actually-loaded model's config and will fail
        # with the real numbers printed if either guess below is wrong --
        # update these two values from that error message on first run.
        n_layers=18,
        d_model=640,
        # Release name as found directly in the SAELens registry by the user
        # of this project -- could not be independently corroborated via
        # search (Google's own docs describe the naming as
        # "gemma-scope-2-270m-pt-resid_post", but SAELens registry aliases
        # aren't all browsable that way). Trusted as given; if SAE.from_pretrained
        # rejects it, check the current SAELens pretrained_saes.yaml directly.
        sae=SAEResource(release="gemma-scope-2-270m-pt-res-all"),
        # sae_id: best guess is the simplest possible form, on the theory that
        # a "-res-all" release (one config per layer, covering every layer)
        # doesn't need the width/L0 qualifiers that Gemma Scope 2's general
        # per-site releases require (e.g. "layer_12_width_16k_l0_medium").
        # If this fails, that fuller format is the documented fallback --
        # check the release's file listing on HuggingFace either way.
        sae_id_template="layer_{layer}_width_16k_l0_small",
        hook_side="post",  # Gemma Scope 2's residual-stream SAEs are trained
        # post-block, same site convention as Pythia's SAE above.
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
}

# Which layer(s) the invariance loss is computed at. THESE ARE PLACEHOLDERS.
# Fill in from Step 1's depth_profile.png -- the layer(s) with the largest
# delta-AUROC (SAE minus raw) for each model -- before running anything for real.
# If two or three adjacent layers are close, list all of them; the loss sums
# over every layer listed.
INVARIANCE_LAYERS = {
    "gpt2-small": [6],           # placeholder
    "pythia-70m-deduped": [3],   # placeholder
    "gemma3-270m": [9],          # placeholder -- doubly so: also depends on n_layers above being correct
}

# space: "raw" or "sae" -- which representation the magnitude term is computed on.
# use_support_term: adds the soft-support (soft-Dice) term; only meaningful for space="sae".
CONDITION_SPECS = {
    "C1_lm_only":               dict(use_invariance=False, space=None,  use_support_term=False),
    "C2_raw_invariance":        dict(use_invariance=True,  space="raw", use_support_term=False),
    "C3_sae_magnitude":         dict(use_invariance=True,  space="sae", use_support_term=False),
    "C4_sae_magnitude_support": dict(use_invariance=True,  space="sae", use_support_term=True),
}

LAMBDA_GRID = [0.1, 1.0, 3.0, 10.0]  # swept for C2-C4; C1 has no invariance term (lambda ignored)

PAWS_CONFIG = dict(
    hf_name="google-research-datasets/paws",
    hf_config="labeled_final",
    random_seed=0,
    n_train_pairs_per_condition=4000,       # same-meaning pairs feed the invariance loss; diff-meaning
                                              # training-split pairs are loaded but never optimized against
    n_validation_pairs_per_condition=300,    # periodic collapse / AUROC-gap check during training
    n_test_pairs_per_condition=1000,          # untouched until evaluate.py -- identical to Step 1's test set
)

WIKITEXT_CONFIG = dict(
    hf_name="Salesforce/wikitext",   # current canonical, parquet-backed location.
    hf_config="wikitext-2-raw-v1",    # The unprefixed "wikitext" repo still exists but has had
    # loading-script issues since HF deprecated script-based dataset loading;
    # Salesforce/wikitext is the maintained mirror as of when this was written.
    max_seq_len=256,
)

TRAIN_CONFIG = dict(
    device="cuda",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    learning_rate=1e-4,
    batch_size_lm=8,
    batch_size_invariance=8,   # number of PAWS pairs per step; processed as one forward
                                # pass per sentence (see hooked_activations.py), not
                                # padded into a single batched call
    max_steps=500,               # deliberately short -- see README on SAE-drift risk
    eval_every=50,
    support_threshold=0.0,
    eval_batch_size=32,           # padded batch size for evaluate.py's forward passes;
                                    # unrelated to batch_size_lm/batch_size_invariance above,
                                    # which are training-step sizes processed unbatched
    output_dir="results/step2_llm",
)
