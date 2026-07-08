"""
WikiText-2 loading and batching for the language-modeling loss term. Kept as a
separate dataset from the PAWS pairs used in the invariance loss, so the two
loss terms are shaped by genuinely different data rather than both being drawn
from the same narrow sentence pool.
"""

import torch
from datasets import load_dataset


def load_wikitext_train_text(hf_name: str, hf_config: str) -> list[str]:
    ds = load_dataset(hf_name, hf_config, split="train")
    # WikiText-2-raw ships one "document" (often a single line) per row, many of
    # which are blank lines or bare section headers left over from the original
    # Wikipedia dump extraction; drop anything too short to be a useful example.
    return [row["text"] for row in ds if len(row["text"].strip()) > 20]


class WikiTextBatcher:
    """
    Tokenizes all provided texts once, concatenates them (separated by the
    tokenizer's EOS token) into a single long token stream, and slices that
    stream into non-overlapping fixed-length blocks. This is the standard
    block-based approach for causal LM training on WikiText -- padding each
    short document individually would waste most of a batch on padding tokens
    for a dataset this granular.
    """

    def __init__(self, texts: list[str], tokenizer, max_seq_len: int, device: str, seed: int = 0):
        self.max_seq_len = max_seq_len
        self.device = device

        eos = tokenizer.eos_token or tokenizer.pad_token or "\n"
        joined_ids: list[int] = []
        for text in texts:
            joined_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            joined_ids.extend(tokenizer(eos, add_special_tokens=False)["input_ids"])

        n_blocks = max(1, len(joined_ids) // max_seq_len)
        joined_ids = joined_ids[: n_blocks * max_seq_len]
        self.blocks = torch.tensor(joined_ids, dtype=torch.long).view(n_blocks, max_seq_len)

        rng = torch.Generator().manual_seed(seed)
        self._order = torch.randperm(self.blocks.size(0), generator=rng)
        self._pos = 0

    def next_batch(self, batch_size: int) -> torch.Tensor:
        batch_size = min(batch_size, len(self._order))
        if self._pos + batch_size > len(self._order):
            self._pos = 0  # simple wraparound; fine for a short PoC run
        idx = self._order[self._pos : self._pos + batch_size]
        self._pos += batch_size
        return self.blocks[idx].to(self.device)  # [batch_size, max_seq_len]
