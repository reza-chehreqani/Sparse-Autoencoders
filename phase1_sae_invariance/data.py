"""
Load PAWS paraphrase / non-paraphrase pairs.

WHY PAWS AND NOT A GENERIC PARAPHRASE DATASET:

PAWS's negative pairs are deliberately constructed to have HIGH LEXICAL
OVERLAP with DIFFERENT meaning (e.g. word-order swaps that change who
did what to whom), while its positive pairs are true paraphrases. This
gives us a controlled experiment: if SAE-feature similarity at a given
layer is high for BOTH groups, that layer is responding to surface
word overlap, not meaning. If SAE-feature similarity is high only for
the true-paraphrase group, that layer has captured genuine semantic
content. A generic paraphrase dataset (e.g. STS-B high-similarity
pairs only) would not let us distinguish these two explanations,
because there would be no lexically-similar-but-different-meaning
control group.
"""

import random
from datasets import load_dataset

from config import Phase1Config


def load_paraphrase_pairs(cfg: Phase1Config):
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    random.seed(cfg.random_seed)

    pos = [r for r in ds if r["label"] == 1]
    neg = [r for r in ds if r["label"] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    pos = pos[: cfg.n_pairs_per_class]
    neg = neg[: cfg.n_pairs_per_class]

    if len(pos) < cfg.n_pairs_per_class or len(neg) < cfg.n_pairs_per_class:
        print(
            f"WARNING: requested {cfg.n_pairs_per_class} pairs per class, "
            f"but PAWS only provided {len(pos)} positive / {len(neg)} negative."
        )

    pairs = []
    for r in pos:
        pairs.append({"sentence_a": r["sentence1"], "sentence_b": r["sentence2"], "is_paraphrase": True})
    for r in neg:
        pairs.append({"sentence_a": r["sentence1"], "sentence_b": r["sentence2"], "is_paraphrase": False})

    random.shuffle(pairs)
    return pairs
