"""Uniform tokenizer interface for residue-level and BPE protein vocabularies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from utils.bpe import AA20, train_protein_bpe

UNK = "X"
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
CLS_TOKEN = "[CLS]"
EOS_TOKEN = "[EOS]"
MLM_SPECIALS = [PAD_TOKEN, MASK_TOKEN, CLS_TOKEN, EOS_TOKEN]
AA20_STR = "".join(AA20)
AA_INDEX = {c: i for i, c in enumerate(AA20_STR)}


class ResidueVocab:
    """Residue-level tokenization: one token per amino acid.

    This is the vocabulary granularity used by residue-level PLMs such as
    ESM-2, whose input alphabet is the 20 standard residues plus specials.
    """

    name = "residue"

    def __init__(self) -> None:
        self.itos = [UNK] + list(AA20)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode_ids(self, seq: str) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(c, unk) for c in seq]

    def token_string(self, token_id: int) -> str:
        return self.itos[token_id]


class BpeVocab:
    """Vanilla character-level BPE over amino-acid strings."""

    name = "bpe"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def train(
        cls,
        sequences: list[str],
        vocab_size: int,
        max_token_length: int = 16,
    ) -> "BpeVocab":
        return cls(
            train_protein_bpe(
                sequences,
                vocab_size=vocab_size,
                max_token_length=max_token_length,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "BpeVocab":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode_ids(self, seq: str) -> list[int]:
        return self.tokenizer.encode(seq).ids

    def token_string(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id)


def build_vocab(
    vocab_size: int,
    train_sequences: list[str],
    *,
    max_token_length: int = 16,
    cache_path: str | Path | None = None,
) -> ResidueVocab | BpeVocab:
    """Residue-level vocabulary when `vocab_size <= 0`, otherwise BPE.

    BPE is always fit on training sequences only, so test sequences never
    influence the vocabulary.
    """
    if vocab_size <= 0:
        return ResidueVocab()

    if cache_path is not None and Path(cache_path).is_file():
        return BpeVocab.load(cache_path)

    vocab = BpeVocab.train(
        train_sequences,
        vocab_size=vocab_size,
        max_token_length=max_token_length,
    )
    if cache_path is not None:
        vocab.save(cache_path)
    return vocab


@dataclass
class EncodedSeq:
    """Token ids plus residue↔token alignment for one protein.

    `ids` includes CLS at the front and EOS at the end. Residue index `i`
    is covered by token position `res2tok[i]`.
    """

    ids: np.ndarray
    res2tok: np.ndarray
    tok_start: np.ndarray
    tok_end: np.ndarray
    sequence: str

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[0])


class MLMVocab:
    """Residue or BPE vocabulary with PAD/MASK/CLS/EOS for masked LMs."""

    def __init__(self, base: ResidueVocab | BpeVocab) -> None:
        self.base = base
        self.name = base.name
        if isinstance(base, ResidueVocab):
            self.itos = list(MLM_SPECIALS) + list(base.itos)
            self.stoi = {t: i for i, t in enumerate(self.itos)}
            self.tokenizer = None
        else:
            tok = base.tokenizer
            existing = tok.get_vocab()
            missing = [s for s in MLM_SPECIALS if s not in existing]
            if missing:
                tok.add_special_tokens(missing)
            self.tokenizer = tok
            self.stoi = tok.get_vocab()
            self.itos = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]

        self.pad_id = self.stoi[PAD_TOKEN]
        self.mask_id = self.stoi[MASK_TOKEN]
        self.cls_id = self.stoi[CLS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.special_ids = {self.pad_id, self.mask_id, self.cls_id, self.eos_id}
        self.contain_table, self.token_aa = self._build_aa_tables()

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def token_string(self, token_id: int) -> str:
        return self.itos[token_id]

    def _build_aa_tables(self) -> tuple[np.ndarray, np.ndarray]:
        max_len = max((len(s) for s in self.itos), default=1)
        contain = np.zeros((self.vocab_size, 20), dtype=bool)
        token_aa = np.full((self.vocab_size, max_len), -1, dtype=np.int16)
        for i, s in enumerate(self.itos):
            if i in self.special_ids:
                continue
            for j, ch in enumerate(s):
                aa = AA_INDEX.get(ch)
                if aa is None:
                    continue
                contain[i, aa] = True
                token_aa[i, j] = aa
        return contain, token_aa

    def encode(self, seq: str) -> EncodedSeq:
        if self.tokenizer is None:
            unk = self.stoi[UNK]
            content = np.array([self.stoi.get(c, unk) for c in seq], dtype=np.int64)
            ids = np.concatenate(
                ([self.cls_id], content, [self.eos_id])
            ).astype(np.int64)
            L = len(seq)
            T = ids.shape[0]
            res2tok = np.arange(L, dtype=np.int64) + 1
            tok_start = np.full(T, -1, dtype=np.int64)
            tok_end = np.full(T, -1, dtype=np.int64)
            tok_start[1:-1] = np.arange(L, dtype=np.int64)
            tok_end[1:-1] = np.arange(L, dtype=np.int64) + 1
            return EncodedSeq(ids=ids, res2tok=res2tok, tok_start=tok_start, tok_end=tok_end, sequence=seq)

        enc = self.tokenizer.encode(seq)
        content_ids = np.array(enc.ids, dtype=np.int64)
        ids = np.concatenate(([self.cls_id], content_ids, [self.eos_id])).astype(np.int64)
        L = len(seq)
        T = ids.shape[0]
        res2tok = np.full(L, -1, dtype=np.int64)
        tok_start = np.full(T, -1, dtype=np.int64)
        tok_end = np.full(T, -1, dtype=np.int64)
        for ti, (s, e) in enumerate(enc.offsets):
            if e <= s:
                continue
            pos = ti + 1
            s = max(0, min(L, s))
            e = max(0, min(L, e))
            res2tok[s:e] = pos
            tok_start[pos] = s
            tok_end[pos] = e
        if np.any(res2tok < 0):
            raise ValueError("BPE offsets did not cover every residue")
        return EncodedSeq(ids=ids, res2tok=res2tok, tok_start=tok_start, tok_end=tok_end, sequence=seq)
