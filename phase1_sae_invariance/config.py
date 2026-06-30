"""
Phase 1 configuration: Layer-Stratified SAE Paraphrase Invariance study.

Edit these values before running. Defaults target Gemma-2-2B with the
Gemma Scope canonical residual-stream SAEs, loaded via `sae_lens`.

Before running the full pipeline, run `list_available_saes.py` once to
confirm the `sae_release` / sae_id format below matches your installed
sae_lens version (this API has changed across versions in the past).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Phase1Config:
    # --- Model ---
    model_name: str = "gemma-2-2b"   # TransformerLens / HookedSAETransformer name
    device: str = "cuda"              # "cuda" or "cpu"

    # --- SAEs (Gemma Scope, canonical residual-stream release) ---
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_width: str = "width_16k"      # canonical 16k-width SAEs exist for every layer
    layers: List[int] = field(default_factory=lambda: list(range(0, 26)))  # gemma-2-2b has 26 layers
    hook_point_template: str = "blocks.{layer}.hook_resid_post"

    # --- Data ---
    dataset_name: str = "google-research-datasets/paws"
    dataset_config: str = "labeled_final"
    dataset_split: str = "test"
    n_pairs_per_class: int = 250      # paraphrase pairs AND non-paraphrase pairs, each
    max_seq_len: int = 64
    random_seed: int = 0

    # --- Feature extraction ---
    pooling: str = "mean"             # "mean" or "last_token"

    # --- Output ---
    output_dir: str = "./phase1_outputs"
