"""
Loads the base causal LM via plain HuggingFace `transformers`, then wraps it
with a LoRA adapter via `peft`. Deliberately not TransformerLens here (see
hooked_activations.py's module docstring for why) -- `AutoModelForCausalLM.
from_pretrained` with default settings loads weights exactly as HuggingFace
stored them, which is the same "no extra processing" activation distribution
Step 1 got from TransformerLens's `from_pretrained_no_processing`, so switching
loaders does not introduce a new SAE/model distribution mismatch.
"""

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[hf_model_loading] CUDA requested but not available -- falling back to CPU.")
        return "cpu"
    return requested


def load_tokenizer(model_cfg):
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _check_model_shape(base_model, model_cfg) -> None:
    cfg = base_model.config
    actual_n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    assert actual_n_layers == model_cfg.n_layers, (
        f"Expected {model_cfg.n_layers} layers for {model_cfg.name}, got {actual_n_layers} -- update config.py."
    )
    actual_d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    assert actual_d_model == model_cfg.d_model, (
        f"Expected d_model={model_cfg.d_model} for {model_cfg.name}, got {actual_d_model} -- update config.py."
    )


def load_lora_model(model_cfg, train_cfg: dict, device: str) -> PeftModel:
    """For TRAINING: attaches a fresh, randomly-initialized LoRA adapter (peft
    zero-initializes the LoRA B matrix by default, so this starts numerically
    identical to the plain pretrained model)."""
    # torch_dtype is explicit and deliberate: without it, from_pretrained defaults
    # to whatever dtype the checkpoint itself declares. gpt2's checkpoint doesn't
    # declare one (so it happens to load as float32); pythia-70m-deduped's does
    # declare float16, which is fine for a large model with plenty of numerical
    # headroom but was empirically confirmed (see debug_pythia_loss.py) to cause
    # occasional catastrophic loss spikes on rare/out-of-distribution token
    # patterns for a model this small. Forcing float32 for both removes this as
    # a source of divergence between the two models' training dynamics.
    base_model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_model_name, torch_dtype=torch.float32).to(device)
    _check_model_shape(base_model, model_cfg)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=model_cfg.lora_target_modules,
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_model_for_eval(model_cfg, device: str, adapter_path: str | None = None):
    """
    For EVALUATION: either attaches a previously-SAVED LoRA adapter (via the
    standard `PeftModel.from_pretrained(base_model, adapter_path)` pattern), or,
    if `adapter_path` is None, returns the plain pretrained model untouched --
    this is the pretrained baseline condition (C0).

    Returns (model, hf_model): `model` is what you call for LM loss / generation
    (a PeftModel if an adapter was loaded, otherwise the plain HF model itself);
    `hf_model` is always the plain underlying HF model, suitable for registering
    activation hooks on (see hooked_activations.py).
    """
    # Same explicit torch_dtype=torch.float32 as load_lora_model, and for the
    # same reason: without it, this would default to whatever dtype the
    # checkpoint declares, which is not a safe assumption to leave implicit
    # per-model (see load_lora_model's comment and debug_pythia_loss.py).
    base_model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_model_name, torch_dtype=torch.float32).to(device)
    _check_model_shape(base_model, model_cfg)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        hf_model = model.get_base_model()
    else:
        model = base_model
        hf_model = base_model

    model.eval()
    return model, hf_model
