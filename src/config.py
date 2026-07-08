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
        hf_model_name="google/gemma-3-270m",
        model_family="gemma3",
        n_layers=18,
        d_model=640,
        sae=SAEResource(release="gemma-scope-2-270m-pt-res-all"),
        sae_id_template="layer_{layer}_width_16k_l0_small",
        hook_side="post",
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
    "gemma3-270m": [9],          # placeholder -- middle of the now-confirmed 18 layers
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
    n_train_pairs_per_condition=4000,       # both same- and diff-meaning training pairs are now
                                              # used: same-meaning pairs feed the attractive term,
                                              # diff-meaning pairs feed the repulsive term (see losses.py)
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
    sae_learning_rate=1e-5,      # only used with --joint_sae; deliberately much smaller than
                                   # learning_rate, since the SAE dictionary is already converged
                                   # and this is a gentle continuation, not training from scratch
    sae_recon_loss_weight=1.0,    # only used with --joint_sae; weight on the SAE's own
                                    # reconstruction-MSE anchor term (frozen_sae.FrozenSAE.
                                    # reconstruction_loss) relative to the rest of the total loss --
                                    # this is what stops the SAE from collapsing to a trivial,
                                    # input-independent encoding under the invariance loss alone
    repulsive_margin=0.5,          # target minimum cosine distance for diff-meaning pairs; the
                                     # repulsive term is zero once a pair is already this far apart
                                     # (see losses.py's module docstring for why a hinge, not an
                                     # unbounded "maximize distance" term)
    batch_size_lm=8,
    batch_size_invariance=8,   # number of PAWS pairs per step, PER condition (same and diff each
                                # get this many); processed as one forward pass per sentence (see
                                # hooked_activations.py), not padded into a single batched call
    max_steps=500,               # deliberately short -- see README on SAE-drift risk
    eval_every=50,
    support_threshold=0.0,
    eval_batch_size=32,           # padded batch size for evaluate.py's forward passes;
                                    # unrelated to batch_size_lm/batch_size_invariance above,
                                    # which are training-step sizes processed unbatched
    output_dir="results",
)
