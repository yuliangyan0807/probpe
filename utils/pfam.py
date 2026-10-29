"""Pfam seed parsing and conserved / control segment extraction."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
GAP_CHARS = set(".-")


@dataclass
class FamilyAlignment:
    accession: str
    name: str
    sequences: dict[str, str]  # seq_id -> aligned sequence (equal length)


@dataclass
class SegmentInstance:
    family: str
    seq_id: str
    sequence: str  # ungapped protein / domain sequence
    start: int  # inclusive, 0-based on ungapped sequence
    end: int  # exclusive
    kind: str  # "conserved" or "control"
    mean_conservation: float

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def segment(self) -> str:
        return self.sequence[self.start : self.end]


def iter_stockholm_families(path: str) -> Iterator[FamilyAlignment]:
    """Stream Pfam-A.seed Stockholm alignments one family at a time."""
    accession = name = None
    seqs: dict[str, list[str]] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("# STOCKHOLM"):
                accession = name = None
                seqs = {}
                continue
            if line.startswith("//"):
                if accession and seqs:
                    joined = {sid: "".join(parts) for sid, parts in seqs.items()}
                    lengths = {len(s) for s in joined.values()}
                    if len(lengths) == 1:
                        yield FamilyAlignment(accession, name or accession, joined)
                accession = name = None
                seqs = {}
                continue
            if line.startswith("#=GF ID"):
                name = line.split(None, 2)[2].strip()
            elif line.startswith("#=GF AC"):
                accession = line.split(None, 2)[2].strip().split(".")[0]
            elif not line or line.startswith("#"):
                continue
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                sid, aln = parts
                seqs.setdefault(sid, []).append(aln.strip())


def _column_gap_ratio(aligned_seqs: list[str], col: int) -> float:
    n = len(aligned_seqs)
    if n == 0:
        return 1.0
    gaps = sum(1 for s in aligned_seqs if s[col] in GAP_CHARS)
    return gaps / n


def _column_aa_probs(aligned_seqs: list[str], col: int) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for s in aligned_seqs:
        ch = s[col].upper()
        if ch in AA20:
            counts[ch] = counts.get(ch, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {aa: c / total for aa, c in counts.items()}


def column_conservation(aligned_seqs: list[str], col: int) -> float:
    """C_j = 1 - H_j / log(20). Returns 0 if column has no amino acids."""
    probs = _column_aa_probs(aligned_seqs, col)
    if not probs:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return 1.0 - entropy / math.log(20.0)


def ungapped_sequence(aligned: str) -> str:
    return "".join(ch.upper() for ch in aligned if ch not in GAP_CHARS)


def alignment_to_seq_index(aligned: str) -> list[int | None]:
    """Map each alignment column to ungapped residue index, or None for gaps."""
    mapping: list[int | None] = []
    idx = 0
    for ch in aligned:
        if ch in GAP_CHARS:
            mapping.append(None)
        else:
            mapping.append(idx)
            idx += 1
    return mapping


def is_standard_protein(seq: str, max_nonstandard_frac: float = 0.02) -> bool:
    if not seq:
        return False
    bad = sum(1 for c in seq if c not in AA20)
    return (bad / len(seq)) <= max_nonstandard_frac


def find_conserved_runs(
    conserved_mask: list[bool],
    min_len: int = 4,
    max_len: int = 12,
) -> list[tuple[int, int]]:
    """Return inclusive-exclusive column-index runs of conserved columns."""
    runs: list[tuple[int, int]] = []
    n = len(conserved_mask)
    i = 0
    while i < n:
        if not conserved_mask[i]:
            i += 1
            continue
        j = i
        while j < n and conserved_mask[j]:
            j += 1
        run_len = j - i
        if run_len >= min_len:
            # Prefer middle window if run is longer than max_len
            if run_len <= max_len:
                runs.append((i, j))
            else:
                start = i + (run_len - max_len) // 2
                runs.append((start, start + max_len))
        i = j
    return runs


def extract_gapfree_instance(
    family: str,
    seq_id: str,
    aligned: str,
    col_start: int,
    col_end: int,
    kind: str,
    mean_conservation: float,
) -> SegmentInstance | None:
    """Map an alignment-column window to a gap-free ungapped segment."""
    window = aligned[col_start:col_end]
    if any(ch in GAP_CHARS for ch in window):
        return None
    if any(ch.upper() not in AA20 for ch in window):
        return None

    mapping = alignment_to_seq_index(aligned)
    positions = [mapping[c] for c in range(col_start, col_end)]
    if any(p is None for p in positions):
        return None
    start = positions[0]
    end = positions[-1] + 1
    # Require contiguous residues (no internal gap that was skipped)
    if end - start != (col_end - col_start):
        return None

    seq = ungapped_sequence(aligned)
    if not is_standard_protein(seq):
        return None
    return SegmentInstance(
        family=family,
        seq_id=seq_id,
        sequence=seq,
        start=start,
        end=end,
        kind=kind,
        mean_conservation=mean_conservation,
    )


def sample_control_window(
    sequence: str,
    length: int,
    forbidden: list[tuple[int, int]],
    conserved_residue_mask: list[bool] | None = None,
    rng: random.Random | None = None,
    max_tries: int = 200,
    target_segment: str | None = None,
    max_composition_l1: float | None = None,
) -> tuple[int, int] | None:
    """Sample a length-matched window that does not overlap forbidden intervals.

    If `target_segment` and `max_composition_l1` are set, prefer windows whose
    amino-acid histogram L1 distance (per residue) is below the threshold;
    otherwise pick the best composition match among valid candidates.
    """
    rng = rng or random.Random()
    n = len(sequence)
    if length > n:
        return None

    def overlaps(a: int, b: int) -> bool:
        for fs, fe in forbidden:
            if a < fe and b > fs:
                return True
        return False

    def composition_l1(a: str, b: str) -> float:
        from collections import Counter

        ca, cb = Counter(a), Counter(b)
        keys = set(ca) | set(cb)
        return sum(abs(ca.get(k, 0) - cb.get(k, 0)) for k in keys) / max(len(a), 1)

    candidates = list(range(0, n - length + 1))
    rng.shuffle(candidates)

    scored: list[tuple[float, int, int]] = []
    for start in candidates:
        end = start + length
        if overlaps(start, end):
            continue
        window = sequence[start:end]
        if any(c not in AA20 for c in window):
            continue
        if conserved_residue_mask is not None:
            frac = sum(conserved_residue_mask[start:end]) / length
            if frac > 0.3:
                continue
        if target_segment is not None:
            dist = composition_l1(window, target_segment)
            if max_composition_l1 is not None and dist > max_composition_l1:
                continue
            scored.append((dist, start, end))
        else:
            scored.append((0.0, start, end))
        if len(scored) >= max_tries:
            break

    if scored:
        scored.sort(key=lambda x: x[0])
        # Among the best composition quartile, sample randomly
        best = scored[0][0]
        top = [s for s in scored if s[0] <= best + 1e-9] or scored[: max(1, len(scored) // 4)]
        _, start, end = rng.choice(top)
        return start, end

    # Fallback: ignore composition / conservation preference
    for start in candidates:
        end = start + length
        if overlaps(start, end):
            continue
        window = sequence[start:end]
        if any(c not in AA20 for c in window):
            continue
        return start, end
    return None


@dataclass
class PreparedExample:
    conserved: SegmentInstance
    control: SegmentInstance


def prepare_family_examples(
    family: FamilyAlignment,
    *,
    min_seqs: int = 20,
    min_aln_len: int = 30,
    max_aln_len: int = 500,
    max_gap_ratio: float = 0.30,
    top_conserved_frac: float = 0.30,
    seg_min_len: int = 4,
    seg_max_len: int = 10,
    max_seq_len: int = 1022,
    seqs_per_family: int = 5,
    max_segments_per_seq: int = 2,
    rng: random.Random | None = None,
) -> list[PreparedExample]:
    """Build conserved/control segment pairs for one Pfam family."""
    rng = rng or random.Random()
    aligned_list = list(family.sequences.values())
    n_seq = len(aligned_list)
    if n_seq < min_seqs:
        return []

    aln_len = len(aligned_list[0])
    if aln_len < min_aln_len or aln_len > max_aln_len:
        return []

    keep_cols = [
        c for c in range(aln_len) if _column_gap_ratio(aligned_list, c) <= max_gap_ratio
    ]
    if len(keep_cols) < seg_min_len:
        return []

    cons_scores = {c: column_conservation(aligned_list, c) for c in keep_cols}
    ranked = sorted(keep_cols, key=lambda c: cons_scores[c], reverse=True)
    n_top = max(1, int(math.ceil(len(keep_cols) * top_conserved_frac)))
    top_set = set(ranked[:n_top])

    # Conserved runs must be consecutive in the original alignment.
    conserved_mask = [(c in top_set) for c in range(aln_len)]
    valid_runs = find_conserved_runs(
        conserved_mask, min_len=seg_min_len, max_len=seg_max_len
    )
    if not valid_runs:
        return []

    seq_ids = list(family.sequences.keys())
    rng.shuffle(seq_ids)

    examples: list[PreparedExample] = []
    used_seqs = 0
    for seq_id in seq_ids:
        if used_seqs >= seqs_per_family:
            break
        aligned = family.sequences[seq_id]
        seq = ungapped_sequence(aligned)
        if len(seq) < seg_min_len or len(seq) > max_seq_len:
            continue
        if not is_standard_protein(seq):
            continue

        mapping = alignment_to_seq_index(aligned)
        residue_cons = [False] * len(seq)
        for c in top_set:
            p = mapping[c]
            if p is not None:
                residue_cons[p] = True

        forbidden: list[tuple[int, int]] = []
        made = 0
        for col_start, col_end in valid_runs:
            if made >= max_segments_per_seq:
                break
            run_cols = [c for c in range(col_start, col_end) if c in cons_scores]
            mean_c = sum(cons_scores[c] for c in run_cols) / max(len(run_cols), 1)
            cons_inst = extract_gapfree_instance(
                family.accession,
                seq_id,
                aligned,
                col_start,
                col_end,
                kind="conserved",
                mean_conservation=mean_c,
            )
            if cons_inst is None:
                continue

            ctrl = sample_control_window(
                seq,
                cons_inst.length,
                forbidden=[(cons_inst.start, cons_inst.end)] + forbidden,
                conserved_residue_mask=residue_cons,
                rng=rng,
            )
            if ctrl is None:
                continue
            c_start, c_end = ctrl
            # Control mean conservation from residue mask proxy
            ctrl_cons = (
                sum(1 for i in range(c_start, c_end) if residue_cons[i]) / cons_inst.length
            )
            ctrl_inst = SegmentInstance(
                family=family.accession,
                seq_id=seq_id,
                sequence=seq,
                start=c_start,
                end=c_end,
                kind="control",
                mean_conservation=ctrl_cons,
            )
            examples.append(PreparedExample(conserved=cons_inst, control=ctrl_inst))
            forbidden.append((cons_inst.start, cons_inst.end))
            forbidden.append((c_start, c_end))
            made += 1

        if made > 0:
            used_seqs += 1

    return examples


def collect_examples(
    seed_path: str,
    *,
    max_families: int = 100,
    seed: int = 0,
    **kwargs,
) -> list[PreparedExample]:
    """Scan Pfam-A.seed and collect prepared conserved/control pairs."""
    rng = random.Random(seed)
    examples: list[PreparedExample] = []
    n_families_used = 0
    for family in iter_stockholm_families(seed_path):
        if n_families_used >= max_families:
            break
        fam_examples = prepare_family_examples(family, rng=rng, **kwargs)
        if not fam_examples:
            continue
        examples.extend(fam_examples)
        n_families_used += 1
    return examples
