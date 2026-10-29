"""ESM-2 scoring helpers for the mutation-coupling preliminary experiment."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _residue_token_ids(tokenizer, sequence: str) -> list[int]:
    """Map each residue to its single-AA token id (excluding special tokens)."""
    ids = []
    for aa in sequence:
        tok_ids = tokenizer.encode(aa, add_special_tokens=False)
        if len(tok_ids) != 1:
            raise ValueError(f"Unexpected tokenization for residue {aa!r}: {tok_ids}")
        ids.append(tok_ids[0])
    return ids


@torch.no_grad()
def masked_log_probs_at_positions(
    model,
    tokenizer,
    sequence: str,
    positions: list[int],
    *,
    batch_size: int = 8,
) -> list[float]:
    """
    For each position i in `positions`, mask only i and return log p(true aa at i).

    Positions are 0-based indices into the ungapped residue sequence.
    """
    if not positions:
        return []

    device = _device_of(model)
    mask_id = tokenizer.mask_token_id
    residue_ids = _residue_token_ids(tokenizer, sequence)
    base_ids = [tokenizer.cls_token_id] + residue_ids + [tokenizer.eos_token_id]

    log_probs: list[float] = []
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        batch = []
        true_ids = []
        logit_index = []
        for pos in chunk:
            ids = list(base_ids)
            ids[pos + 1] = mask_id
            batch.append(ids)
            true_ids.append(residue_ids[pos])
            logit_index.append(pos + 1)

        input_ids = torch.tensor(batch, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        for b, (t_idx, true_id) in enumerate(zip(logit_index, true_ids)):
            lp = F.log_softmax(logits[b, t_idx].float(), dim=-1)[true_id].item()
            log_probs.append(lp)

    return log_probs


def mutation_coupling(
    model,
    tokenizer,
    sequence: str,
    start: int,
    end: int,
    *,
    n_mutations: int = 3,
    batch_size: int = 8,
    rng=None,
) -> dict[str, float]:
    """
    Couple strength inside a span: mutate one span residue, measure PLL drop
    at the other span positions.
    """
    import random as _random

    rng = rng or _random.Random(0)
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    positions = list(range(start, end))
    length = len(positions)
    if length < 2:
        return {
            "coupling": float("nan"),
            "n_mutations_used": 0.0,
            "length": float(length),
        }

    base_lps = masked_log_probs_at_positions(
        model, tokenizer, sequence, positions, batch_size=batch_size
    )
    base_map = {p: lp for p, lp in zip(positions, base_lps)}

    mut_sites = positions[:]
    rng.shuffle(mut_sites)
    mut_sites = mut_sites[: min(n_mutations, length)]

    drops = []
    for j in mut_sites:
        orig = sequence[j]
        choices = [a for a in aa_list if a != orig]
        new_aa = rng.choice(choices)
        mut_seq = sequence[:j] + new_aa + sequence[j + 1 :]
        others = [p for p in positions if p != j]
        mut_lps = masked_log_probs_at_positions(
            model, tokenizer, mut_seq, others, batch_size=batch_size
        )
        for p, lp in zip(others, mut_lps):
            drops.append(base_map[p] - lp)

    return {
        "coupling": float(sum(drops) / len(drops)) if drops else float("nan"),
        "n_mutations_used": float(len(mut_sites)),
        "n_pair_drops": float(len(drops)),
        "frac_positive_drops": float(np.mean([d > 0 for d in drops])) if drops else float("nan"),
        "length": float(length),
    }
