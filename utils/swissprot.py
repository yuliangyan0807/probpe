"""AFDB Swiss-Prot PDB streaming and length-stratified sampling."""

from __future__ import annotations

import gzip
import os
import random
import time
from pathlib import Path

import numpy as np

from utils.alphabet import AA20, AA3TO1

DEFAULT_AFDB = "/mnt/data/dataset/afdb/swissprot"


def accession_from_afdb_name(name: str) -> str:
    if name.startswith("AF-") and "-F1-" in name:
        return name[3 : name.index("-F1-")]
    return Path(name).stem


def parse_afdb_pdb_gz(path: Path, min_len: int, max_len: int):
    """Read N/CA/C and the ATOM sequence from one AFDB Swiss-Prot gzip PDB."""
    n_xyz, ca_xyz, c_xyz, seq = {}, {}, {}, {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            res3 = line[17:20].strip()
            aa = AA3TO1.get(res3)
            if aa is None:
                continue
            resseq = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
            if atom == "N":
                n_xyz[resseq] = xyz
                seq[resseq] = aa
            elif atom == "CA":
                ca_xyz[resseq] = xyz
            elif atom == "C":
                c_xyz[resseq] = xyz
    keys = sorted(set(n_xyz) & set(ca_xyz) & set(c_xyz) & set(seq))
    if len(keys) < min_len:
        return None
    sequence = "".join(seq[k] for k in keys)
    if not (min_len <= len(sequence) <= max_len):
        return None
    if any(c not in AA20 for c in sequence):
        return None
    return (
        accession_from_afdb_name(path.name),
        sequence,
        np.stack([n_xyz[k] for k in keys]),
        np.stack([ca_xyz[k] for k in keys]),
        np.stack([c_xyz[k] for k in keys]),
        path,
    )


def sample_swissprot(
    afdb_dir: str | Path,
    n: int,
    *,
    min_len: int,
    max_len: int,
    seed: int,
    n_bins: int = 4,
) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, Path]]:
    """Length-stratified sample of AFDB Swiss-Prot models."""
    afdb = Path(afdb_dir)
    names = [e.name for e in os.scandir(afdb) if e.name.endswith(".pdb.gz") and e.name.startswith("AF-")]
    if not names:
        raise FileNotFoundError(f"no AF-*.pdb.gz under {afdb}")
    rng = random.Random(seed)
    rng.shuffle(names)
    bins: list[list] = [[] for _ in range(n_bins)]
    per = max(1, n // n_bins)
    scanned = 0
    t0 = time.time()
    for name in names:
        scanned += 1
        rec = parse_afdb_pdb_gz(afdb / name, min_len, max_len)
        if rec is None:
            continue
        length = len(rec[1])
        b = min(n_bins - 1, int((length - min_len) / max(max_len - min_len, 1) * n_bins))
        if len(bins[b]) < per:
            bins[b].append(rec)
        if all(len(x) >= per for x in bins):
            break
        if scanned % 2000 == 0:
            kept = sum(len(x) for x in bins)
            print(f"  scanned {scanned} AFDB files, kept {kept}/{n} ({time.time() - t0:.0f}s)", flush=True)
    out = [x for bucket in bins for x in bucket]
    rng.shuffle(out)
    out = out[:n]
    if len(out) < n:
        print(f"  warning: only {len(out)} Swiss-Prot models in length window", flush=True)
    return out
