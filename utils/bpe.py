"""Vanilla BPE training and token-span utilities for protein sequences."""

from __future__ import annotations

import random
from dataclasses import dataclass

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class TokenSpan:
    start: int
    end: int
    token: str
    n_bpe_tokens: int = 1

    @property
    def length(self) -> int:
        return self.end - self.start


def train_protein_bpe(
    sequences: list[str],
    vocab_size: int = 800,
    max_token_length: int = 16,
) -> Tokenizer:
    """
    Train BPE on raw amino-acid strings.

    No character-isolated pretokenizer: BPE only merges within pretokens.
    """
    clean = [s for s in sequences if s]
    if not clean:
        raise ValueError("No sequences to train BPE")

    vocab_size = max(vocab_size, len(AA20) + 2)
    tokenizer = Tokenizer(BPE(unk_token="X"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["X"],
        initial_alphabet=AA20,
        max_token_length=max_token_length,
        show_progress=False,
    )
    tokenizer.train_from_iterator(clean, trainer=trainer)
    return tokenizer


def tokenize_spans(tokenizer: Tokenizer, sequence: str) -> list[TokenSpan]:
    """Return contiguous residue spans for each BPE token."""
    enc = tokenizer.encode(sequence)
    spans: list[TokenSpan] = []
    for tok, (s, e) in zip(enc.tokens, enc.offsets):
        if e <= s:
            continue
        spans.append(TokenSpan(start=s, end=e, token=sequence[s:e], n_bpe_tokens=1))
    return spans


def find_bpe_aligned_span(
    tokenizer: Tokenizer,
    sequence: str,
    length: int,
    forbidden: list[tuple[int, int]],
    *,
    rng: random.Random | None = None,
) -> TokenSpan | None:
    """
    Find a length-matched span whose boundaries align with BPE tokens.

    Prefers spans covered by fewer BPE tokens (more 'atomic' under BPE).
    """

    def overlaps(a: int, b: int) -> bool:
        for fs, fe in forbidden:
            if a < fe and b > fs:
                return True
        return False

    units = tokenize_spans(tokenizer, sequence)
    if not units:
        return None

    candidates: list[TokenSpan] = []
    for i in range(len(units)):
        total = 0
        for j in range(i, len(units)):
            total += units[j].length
            if total > length:
                break
            if total == length:
                start = units[i].start
                end = units[j].end
                if overlaps(start, end):
                    break
                piece = sequence[start:end]
                if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in piece):
                    candidates.append(
                        TokenSpan(
                            start=start,
                            end=end,
                            token=piece,
                            n_bpe_tokens=j - i + 1,
                        )
                    )
                break

    if not candidates:
        return None

    # Prefer fewer BPE pieces, then central position
    candidates.sort(
        key=lambda sp: (
            sp.n_bpe_tokens,
            abs((sp.start + sp.end) / 2 - len(sequence) / 2),
        )
    )
    # Among equally atomic, optionally sample from the best tier
    best_n = candidates[0].n_bpe_tokens
    top = [c for c in candidates if c.n_bpe_tokens == best_n]
    if rng is not None:
        return rng.choice(top)
    return top[0]


# Back-compat alias
def find_bpe_span(*args, **kwargs):
    return find_bpe_aligned_span(*args, **kwargs)
