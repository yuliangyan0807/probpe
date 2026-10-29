"""UniRef FASTA streaming, filtering, and train/test splitting.

The UniRef FASTA releases are sorted by sequence length in descending order, so
reading a prefix of the file yields a sample of extremely long proteins rather
than a representative one. All sampling here therefore streams the whole archive
and keeps a reservoir sample of the sequences that pass the length and alphabet
filters.
"""

from __future__ import annotations

import gzip
import json
import random
import time
from pathlib import Path
from typing import Iterator

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_BYTES = frozenset(AA20.encode())

DEFAULT_UNIREF50 = "/mnt/data/dataset/uniprot/uniref50.fasta.gz"


def iter_fasta_records(path: str, buffer_size: int = 1 << 22) -> Iterator[bytes]:
    """Stream raw FASTA records (header line plus sequence lines) as bytes.

    Chunked byte-level splitting is several times faster than decoding the
    archive line by line, which matters when scanning tens of gigabytes.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        tail = b""
        while True:
            chunk = handle.read(buffer_size)
            if not chunk:
                break
            parts = (tail + chunk).split(b"\n>")
            tail = parts.pop()
            for part in parts:
                yield part
        if tail:
            yield tail


def record_sequence(record: bytes) -> bytes:
    """Concatenate the sequence lines of a raw FASTA record."""
    newline = record.find(b"\n")
    if newline < 0:
        return b""
    return record[newline + 1 :].replace(b"\n", b"")


def reservoir_sample_sequences(
    path: str,
    *,
    n_sequences: int,
    min_len: int = 40,
    max_len: int = 512,
    seed: int = 0,
    scan_limit: int | None = None,
    report_every: int = 5_000_000,
) -> list[str]:
    """Uniform sample of clean sequences from a full pass over the archive."""
    rng = random.Random(seed)
    reservoir: list[bytes] = []
    n_seen = 0
    n_records = 0
    started = time.time()

    for record in iter_fasta_records(path):
        n_records += 1
        if scan_limit is not None and n_records > scan_limit:
            break

        seq = record_sequence(record)
        if not (min_len <= len(seq) <= max_len):
            continue
        if not AA20_BYTES.issuperset(seq):
            continue

        n_seen += 1
        if len(reservoir) < n_sequences:
            reservoir.append(seq)
        else:
            j = rng.randrange(n_seen)
            if j < n_sequences:
                reservoir[j] = seq

        if report_every and n_seen % report_every == 0:
            print(
                f"  scanned {n_records:,} records, kept {n_seen:,} candidates "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

    if len(reservoir) < n_sequences:
        raise ValueError(
            f"only {len(reservoir)} clean sequences found in {path}; need {n_sequences}"
        )

    print(
        f"  scan finished: {n_records:,} records, {n_seen:,} candidates, "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    rng.shuffle(reservoir)
    return [s.decode("ascii") for s in reservoir]


def _write_lines(path: Path, seqs: list[str]) -> None:
    path.write_text("\n".join(seqs) + "\n", encoding="utf-8")


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln]


def load_split(
    *,
    fasta_path: str = DEFAULT_UNIREF50,
    cache_dir: str = "data/uniref50_split",
    n_train: int = 200_000,
    n_test: int = 20_000,
    min_len: int = 40,
    max_len: int = 512,
    scan_limit: int | None = None,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Build (or reuse) a disjoint train/test split of UniRef50 sequences.

    UniRef50 clusters share at most 50% sequence identity, so a random split
    already keeps train and test sequences far apart in sequence space.
    """
    cache = Path(cache_dir)
    meta_path = cache / "meta.json"
    train_path = cache / "train.txt"
    test_path = cache / "test.txt"

    meta = {
        "fasta_path": str(fasta_path),
        "n_train": n_train,
        "n_test": n_test,
        "min_len": min_len,
        "max_len": max_len,
        "scan_limit": scan_limit,
        "seed": seed,
    }

    if meta_path.is_file() and train_path.is_file() and test_path.is_file():
        if json.loads(meta_path.read_text(encoding="utf-8")) == meta:
            return _read_lines(train_path), _read_lines(test_path)

    seqs = reservoir_sample_sequences(
        fasta_path,
        n_sequences=n_train + n_test,
        min_len=min_len,
        max_len=max_len,
        seed=seed,
        scan_limit=scan_limit,
    )
    train, test = seqs[:n_train], seqs[n_train : n_train + n_test]

    cache.mkdir(parents=True, exist_ok=True)
    _write_lines(train_path, train)
    _write_lines(test_path, test)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return train, test
