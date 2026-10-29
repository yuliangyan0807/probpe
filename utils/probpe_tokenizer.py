"""Serializable, reversible tokenizer for ordered ProBPE merge rules."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Iterable, Sequence

from utils.alphabet import AA20

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
EOS_TOKEN = "[EOS]"
MASK_TOKEN = "[MASK]"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, EOS_TOKEN, MASK_TOKEN)


class ProBPETokenizer:
    """BPE tokenizer with an LLM-style id interface and special tokens.

    The ordered merge list is the tokenizer model. Encoding uses a heap-based
    implementation of ranked BPE, so it is equivalent to replaying every rule
    in order but does not scan the sequence once per rule.
    """

    FORMAT = "probpe-tokenizer-v1"

    def __init__(
        self,
        vocab: Sequence[str],
        merges: Sequence[tuple[str, str]],
        *,
        metadata: dict | None = None,
    ) -> None:
        content_vocab = list(vocab)
        if len(content_vocab) != len(set(content_vocab)):
            raise ValueError("content vocabulary contains duplicate tokens")
        if any(token in SPECIAL_TOKENS for token in content_vocab):
            raise ValueError("special tokens must not appear in the content vocabulary")
        missing_amino_acids = [aa for aa in AA20 if aa not in content_vocab]
        if missing_amino_acids:
            raise ValueError(f"content vocabulary is missing amino acids: {missing_amino_acids}")

        self.special_tokens = list(SPECIAL_TOKENS)
        self.content_vocab = content_vocab
        self.vocab = self.special_tokens + self.content_vocab
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = list(self.vocab)
        self.merges = [(str(u), str(v)) for u, v in merges]
        self.metadata = dict(metadata or {})

        self.merge_rank: dict[tuple[str, str], int] = {}
        self.merge_result: dict[tuple[str, str], str] = {}
        known = set(AA20)
        for rank, (u, v) in enumerate(self.merges):
            if u not in known or v not in known:
                raise ValueError(f"merge {rank} references an unknown child: {u!r} + {v!r}")
            pair = (u, v)
            if pair in self.merge_rank:
                raise ValueError(f"duplicate merge pair at rank {rank}: {pair}")
            w = u + v
            if w not in self.token_to_id:
                raise ValueError(f"merge result is absent from vocabulary at rank {rank}: {w!r}")
            self.merge_rank[pair] = rank
            self.merge_result[pair] = w
            known.add(w)

        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.cls_token = CLS_TOKEN
        self.eos_token = EOS_TOKEN
        self.mask_token = MASK_TOKEN
        self.pad_token_id = self.token_to_id[PAD_TOKEN]
        self.unk_token_id = self.token_to_id[UNK_TOKEN]
        self.cls_token_id = self.token_to_id[CLS_TOKEN]
        self.eos_token_id = self.token_to_id[EOS_TOKEN]
        self.mask_token_id = self.token_to_id[MASK_TOKEN]
        self.special_token_ids = set(range(len(self.special_tokens)))

    def __len__(self) -> int:
        return len(self.vocab)

    def _push_pair(
        self,
        heap: list[tuple[int, int, int]],
        token: list[str],
        previous: list[int],
        following: list[int],
        alive: list[bool],
        left: int,
    ) -> None:
        if left < 0 or not alive[left]:
            return
        right = following[left]
        if right < 0 or not alive[right]:
            return
        rank = self.merge_rank.get((token[left], token[right]))
        if rank is not None:
            heapq.heappush(heap, (rank, left, right))

    def tokenize(self, sequence: str) -> list[str]:
        """Tokenize an amino-acid sequence without adding special tokens."""
        if not sequence:
            return []
        invalid = sorted(set(sequence) - set(AA20))
        if invalid:
            raise ValueError(f"sequence contains non-canonical residues: {invalid}")

        token = list(sequence)
        n = len(token)
        previous = [i - 1 for i in range(n)]
        following = [i + 1 for i in range(n)]
        following[-1] = -1
        alive = [True] * n
        heap: list[tuple[int, int, int]] = []
        for left in range(n - 1):
            self._push_pair(heap, token, previous, following, alive, left)

        while heap:
            rank, left, right = heapq.heappop(heap)
            if not alive[left] or not alive[right] or following[left] != right:
                continue
            pair = (token[left], token[right])
            if self.merge_rank.get(pair) != rank:
                continue

            token[left] = self.merge_result[pair]
            alive[right] = False
            next_right = following[right]
            following[left] = next_right
            if next_right >= 0:
                previous[next_right] = left
            self._push_pair(heap, token, previous, following, alive, previous[left])
            self._push_pair(heap, token, previous, following, alive, left)

        output: list[str] = []
        index = 0
        while index >= 0:
            if alive[index]:
                output.append(token[index])
            index = following[index]
        return output

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id[token] for token in self.tokenize(sequence)]
        if add_special_tokens:
            return [self.cls_token_id, *ids, self.eos_token_id]
        return ids

    def encode_batch(
        self,
        sequences: Iterable[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        return [self.encode(sequence, add_special_tokens=add_special_tokens) for sequence in sequences]

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        pieces: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id < 0 or token_id >= len(self.id_to_token):
                raise ValueError(f"token id out of range: {token_id}")
            if skip_special_tokens and token_id in self.special_token_ids:
                continue
            pieces.append(self.id_to_token[token_id])
        return "".join(pieces)

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[int(token_id)] for token_id in ids]

    def convert_tokens_to_ids(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

    def to_dict(self) -> dict:
        return {
            "format": self.FORMAT,
            "special_tokens": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "cls_token": self.cls_token,
                "eos_token": self.eos_token,
                "mask_token": self.mask_token,
            },
            "model": {
                "type": "BPE",
                "content_vocab": self.content_vocab,
                "merges": [[u, v] for u, v in self.merges],
            },
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ProBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.FORMAT:
            raise ValueError(f"unsupported tokenizer format: {payload.get('format')!r}")
        model = payload["model"]
        return cls(
            model["content_vocab"],
            [tuple(pair) for pair in model["merges"]],
            metadata=payload.get("metadata", {}),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or use a saved ProBPE tokenizer")
    parser.add_argument("tokenizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--encode", metavar="SEQUENCE")
    group.add_argument("--decode", metavar="IDS", help="Comma-separated token ids")
    args = parser.parse_args()

    tokenizer = ProBPETokenizer.load(args.tokenizer)
    if args.encode is not None:
        ids = tokenizer.encode(args.encode)
        print(json.dumps({"ids": ids, "tokens": tokenizer.convert_ids_to_tokens(ids)}))
    else:
        ids = [int(value) for value in args.decode.split(",") if value.strip()]
        print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
