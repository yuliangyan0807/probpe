"""Small randomly initialized ESM-style MLM, with residue-centric test scoring.

Training is ordinary token-level masked LM: 15% of *tokens* are masked and the
model predicts the original token.

Test masking is chosen on *residues*, then mapped onto tokens:

- Residue vocabulary: masking a residue is masking that token; the 20-way
  prediction is the usual amino-acid MLM.
- BPE vocabulary: a residue sits inside a multi-residue token, so the input
  mask replaces that whole covering token (the model has no residue-level
  embedding). The model still predicts a token. A prediction counts as
  correct when the predicted token *contains* the held-out amino acid.

The containment rule is the metric that makes the two vocabularies comparable.
It is also optimistic for long tokens, so we additionally report the chance
rate (train-unigram probability that a random token contains the amino acid)
and an aligned check (the character at the residue's offset inside the token).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import EsmConfig, EsmForMaskedLM

from utils.seq_vocab import AA_INDEX, EncodedSeq, MLMVocab


def build_mlm(
    vocab: MLMVocab,
    *,
    hidden_size: int,
    n_layers: int,
    n_heads: int,
    max_positions: int = 1026,
) -> EsmForMaskedLM:
    if hidden_size % n_heads != 0:
        raise ValueError(f"hidden_size {hidden_size} must divide n_heads {n_heads}")
    config = EsmConfig(
        vocab_size=vocab.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=4 * hidden_size,
        max_position_embeddings=max_positions,
        pad_token_id=vocab.pad_id,
        mask_token_id=vocab.mask_id,
        position_embedding_type="rotary",
        token_dropout=True,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        emb_layer_norm_before=False,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        is_folding_model=False,
    )
    return EsmForMaskedLM(config)


def embedding_params(model: EsmForMaskedLM) -> int:
    return int(model.esm.embeddings.word_embeddings.weight.numel())


def dim_for_vocab(
    vocab_size: int,
    *,
    budget: int,
    dim_min: int = 16,
    dim_max: int = 512,
) -> int:
    """Hidden width that keeps `vocab_size * dim` close to `budget`."""
    dim = int(round(budget / max(vocab_size, 1)))
    return max(dim_min, min(dim_max, dim))


@dataclass
class MLMCorpus:
    encoded: list[EncodedSeq]

    @property
    def n_sequences(self) -> int:
        return len(self.encoded)

    @property
    def n_tokens(self) -> int:
        return int(sum(e.n_tokens for e in self.encoded))

    @property
    def n_residues(self) -> int:
        return int(sum(e.n_residues for e in self.encoded))

    @property
    def residues_per_token(self) -> float:
        content = sum(max(e.n_tokens - 2, 0) for e in self.encoded)
        return self.n_residues / max(content, 1)


def encode_corpus(vocab: MLMVocab, sequences: list[str]) -> MLMCorpus:
    encoded = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        encoded.append(vocab.encode(seq))
    return MLMCorpus(encoded=encoded)


def collate_mlm(
    corpus: MLMCorpus,
    indices: list[int],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [corpus.encoded[i].n_tokens for i in indices]
    width = max(lengths)
    ids = np.full((len(indices), width), pad_id, dtype=np.int64)
    attn = np.zeros((len(indices), width), dtype=np.int64)
    for row, idx in enumerate(indices):
        seq = corpus.encoded[idx].ids
        ids[row, : len(seq)] = seq
        attn[row, : len(seq)] = 1
    return torch.from_numpy(ids), torch.from_numpy(attn)


def bert_mask(
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_id: int,
    pad_id: int,
    cls_id: int,
    eos_id: int,
    vocab_size: int,
    rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard BERT masking over content tokens (not specials/padding)."""
    special = (
        (ids == pad_id)
        | (ids == cls_id)
        | (ids == eos_id)
        | (attention_mask == 0)
    )
    candidates = ~special
    mask_pos = (torch.rand(ids.shape, device=ids.device) < rate) & candidates
    labels = ids.clone()
    labels[~mask_pos] = -100

    masked = ids.clone()
    draw = torch.rand(ids.shape, device=ids.device)
    replace_mask = mask_pos & (draw < 0.8)
    random_pos = mask_pos & (draw >= 0.8) & (draw < 0.9)
    masked[replace_mask] = mask_id
    n_rand = int(random_pos.sum())
    if n_rand:
        # Draw from the full vocab; specials are rare and never appear as labels.
        masked[random_pos] = torch.randint(0, vocab_size, (n_rand,), device=ids.device)
    return masked, labels


def sample_residue_masks(
    sequences: list[str],
    *,
    rate: float,
    seed: int,
) -> list[list[int]]:
    """Same residue indices for every vocabulary, so test sites are matched."""
    rng = random.Random(seed)
    out: list[list[int]] = []
    for seq in sequences:
        n = min(len(seq), max(1, int(round(len(seq) * rate))))
        out.append(sorted(rng.sample(range(len(seq)), n)))
    return out


def contain_chance(
    vocab: MLMVocab,
    train: MLMCorpus,
    test_sequences: list[str],
    residue_masks: list[list[int]],
) -> float:
    """Unigram probability that a random content token contains the target AA."""
    counts = np.zeros(vocab.vocab_size, dtype=np.float64)
    for enc in train.encoded:
        content = enc.ids[1:-1]
        counts += np.bincount(content, minlength=vocab.vocab_size)
    counts[list(vocab.special_ids)] = 0.0
    probs = counts / max(counts.sum(), 1.0)
    contain_p = vocab.contain_table.T @ probs  # [20]
    hits = 0.0
    n = 0
    for seq, sites in zip(test_sequences, residue_masks):
        for i in sites:
            aa = AA_INDEX.get(seq[i])
            if aa is None:
                continue
            hits += float(contain_p[aa])
            n += 1
    return hits / max(n, 1)


@dataclass
class MLMEval:
    token_loss: float
    aa_nll: float
    aa_acc: float
    aligned_acc: float
    contain_chance: float
    n_eval_residues: int
    n_masked_tokens: int


def train_mlm(
    model: EsmForMaskedLM,
    corpus: MLMCorpus,
    vocab: MLMVocab,
    *,
    epochs: int,
    lr: float,
    positions_per_batch: int,
    mask_rate: float,
    device: torch.device,
    seed: int = 0,
    progress: bool = False,
) -> list[float]:
    rng = random.Random(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_batches = max(
        len(make_batches(corpus, positions_per_batch=positions_per_batch, shuffle=False)),
        1,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max(n_batches * epochs, 1),
        pct_start=0.1,
    )

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        batches = make_batches(
            corpus, positions_per_batch=positions_per_batch, shuffle=True, rng=rng
        )
        iterator = batches
        if progress:
            from tqdm import tqdm

            iterator = tqdm(batches, desc=f"epoch {epoch + 1}/{epochs}", leave=False)

        running = 0.0
        n_labelled = 0
        for indices in iterator:
            ids, attn = collate_mlm(corpus, indices, vocab.pad_id)
            ids = ids.to(device)
            attn = attn.to(device)
            masked, labels = bert_mask(
                ids,
                attn,
                mask_id=vocab.mask_id,
                pad_id=vocab.pad_id,
                cls_id=vocab.cls_id,
                eos_id=vocab.eos_id,
                vocab_size=vocab.vocab_size,
                rate=mask_rate,
            )
            if (labels != -100).sum() == 0:
                continue
            out = model(input_ids=masked, attention_mask=attn, labels=labels)
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            n = int((labels != -100).sum())
            running += float(loss.detach()) * n
            n_labelled += n
        epoch_losses.append(running / max(n_labelled, 1))
    return epoch_losses


@torch.no_grad()
def evaluate_mlm(
    model: EsmForMaskedLM,
    corpus: MLMCorpus,
    vocab: MLMVocab,
    residue_masks: list[list[int]],
    *,
    positions_per_batch: int,
    device: torch.device,
    chance: float,
) -> MLMEval:
    """Mask covering tokens of sampled residues; score amino-acid recovery."""
    model.to(device).eval()
    contain = torch.from_numpy(vocab.contain_table).to(device)
    token_aa = torch.from_numpy(vocab.token_aa.astype(np.int64)).to(device)

    aa_nll_sum = 0.0
    aa_hits = 0
    aligned_hits = 0
    n_eval = 0
    n_masked_tokens = 0
    token_nll_sum = 0.0
    n_token_labels = 0

    batches = make_batches(corpus, positions_per_batch=positions_per_batch, shuffle=False)
    for indices in batches:
        ids, attn = collate_mlm(corpus, indices, vocab.pad_id)
        ids = ids.to(device)
        attn = attn.to(device)
        masked = ids.clone()
        token_labels = torch.full_like(ids, -100)

        eval_b: list[int] = []
        eval_t: list[int] = []
        eval_aa: list[int] = []
        eval_off: list[int] = []
        for row, idx in enumerate(indices):
            enc = corpus.encoded[idx]
            seq = enc.sequence
            sites = residue_masks[idx]
            seen_tok: set[int] = set()
            for res_i in sites:
                if res_i >= len(seq):
                    continue
                aa = AA_INDEX.get(seq[res_i])
                if aa is None:
                    continue
                tok_pos = int(enc.res2tok[res_i])
                start = int(enc.tok_start[tok_pos])
                offset = res_i - start
                masked[row, tok_pos] = vocab.mask_id
                if tok_pos not in seen_tok:
                    token_labels[row, tok_pos] = ids[row, tok_pos]
                    seen_tok.add(tok_pos)
                    n_masked_tokens += 1
                eval_b.append(row)
                eval_t.append(tok_pos)
                eval_aa.append(aa)
                eval_off.append(offset)

        if not eval_b:
            continue

        logits = model(input_ids=masked, attention_mask=attn).logits
        log_probs = F.log_softmax(logits, dim=-1)

        labelled = token_labels != -100
        if labelled.any():
            token_nll_sum += float(
                F.cross_entropy(
                    logits[labelled], token_labels[labelled], reduction="sum"
                )
            )
            n_token_labels += int(labelled.sum())

        b = torch.tensor(eval_b, device=device)
        t = torch.tensor(eval_t, device=device)
        aa = torch.tensor(eval_aa, device=device)
        off = torch.tensor(eval_off, device=device)
        lp = log_probs[b, t]  # [N, V]
        pred = lp.argmax(dim=-1)

        contain_lp = torch.logsumexp(lp.masked_fill(~contain[:, aa].T, -1e9), dim=-1)
        aa_nll_sum += float((-contain_lp).sum())
        aa_hits += int(contain[pred, aa].sum())

        # Aligned character at the residue offset; tokens shorter than offset fail.
        max_off = token_aa.shape[1]
        valid_off = (off >= 0) & (off < max_off)
        aligned = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
        if valid_off.any():
            aligned[valid_off] = token_aa[pred[valid_off], off[valid_off]] == aa[valid_off]
        aligned_hits += int(aligned.sum())
        n_eval += pred.shape[0]

    return MLMEval(
        token_loss=token_nll_sum / max(n_token_labels, 1),
        aa_nll=aa_nll_sum / max(n_eval, 1),
        aa_acc=aa_hits / max(n_eval, 1),
        aligned_acc=aligned_hits / max(n_eval, 1),
        contain_chance=chance,
        n_eval_residues=n_eval,
        n_masked_tokens=n_masked_tokens,
    )


class _BatchAdapter:
    """Length-only view of an MLM corpus for length-bucketed batching."""

    def __init__(self, corpus: MLMCorpus) -> None:
        self.token_ids = [e.ids for e in corpus.encoded]

    @property
    def n_sequences(self) -> int:
        return len(self.token_ids)


def make_batches(corpus, *, positions_per_batch: int, shuffle: bool = True, rng=None):
    """Group sequences of similar length so each batch holds ~N token positions."""
    if isinstance(corpus, MLMCorpus):
        corpus = _BatchAdapter(corpus)
    order = sorted(range(corpus.n_sequences), key=lambda i: len(corpus.token_ids[i]))
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for idx in order:
        longest = max(longest, len(corpus.token_ids[idx]))
        if current and longest * (len(current) + 1) > positions_per_batch:
            batches.append(current)
            current = [idx]
            longest = len(corpus.token_ids[idx])
        else:
            current.append(idx)
    if current:
        batches.append(current)
    if shuffle:
        (rng or random).shuffle(batches)
    return batches
