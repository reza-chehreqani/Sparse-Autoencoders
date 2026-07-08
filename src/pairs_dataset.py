"""
Builds same-meaning / diff-meaning PAWS pairs for a given split. Same dataset and
contrast logic as Step 1 (see that project's pairs_dataset.py for the full
rationale for using PAWS instead of a generated attack) -- duplicated here rather
than imported cross-directory so Step 2 can be run independently of Step 1's
folder layout.

The one difference from Step 1: this loads one split at a time (train,
validation, or test) with its own sample size, since Step 2 needs all three for
different purposes (train the invariance loss, monitor during training, and
evaluate at the end), whereas Step 1 only ever needed the test split.
"""

import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class SentencePair:
    sentence_a: str
    sentence_b: str
    same_meaning: bool
    token_jaccard: float


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def load_paws_split(
    hf_name: str,
    hf_config: str,
    split: str,
    n_pairs_per_condition: int,
    random_seed: int,
) -> list[SentencePair]:
    ds = load_dataset(hf_name, hf_config, split=split)

    same_meaning_rows = [r for r in ds if r["label"] == 1]
    diff_meaning_rows = [r for r in ds if r["label"] == 0]

    rng = random.Random(random_seed)
    rng.shuffle(same_meaning_rows)
    rng.shuffle(diff_meaning_rows)

    same_meaning_rows = same_meaning_rows[:n_pairs_per_condition]
    diff_meaning_rows = diff_meaning_rows[:n_pairs_per_condition]

    pairs: list[SentencePair] = []
    for r in same_meaning_rows:
        pairs.append(
            SentencePair(r["sentence1"], r["sentence2"], True, _token_jaccard(r["sentence1"], r["sentence2"]))
        )
    for r in diff_meaning_rows:
        pairs.append(
            SentencePair(r["sentence1"], r["sentence2"], False, _token_jaccard(r["sentence1"], r["sentence2"]))
        )
    return pairs
