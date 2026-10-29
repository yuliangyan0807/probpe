"""Shared matched biological / BPE / random span triplets for preliminary experiments."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from utils.pfam import (
    AA20,
    collect_examples,
    iter_stockholm_families,
    is_standard_protein,
    sample_control_window,
    ungapped_sequence,
)
from utils.bpe import find_bpe_aligned_span, train_protein_bpe, tokenize_spans

DEFAULT_SEED = "/mnt/data/dataset/pfam/Pfam-A.seed"
DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"


@dataclass
class SpanItem:
    family: str
    seq_id: str
    sequence: str
    start: int
    end: int
    kind: str  # biological | bpe | random
    segment: str
    group_id: str
    n_bpe_tokens: int = 1
    mean_conservation: float = 0.0
    composition_l1_to_bio: float = 0.0


def composition_l1(a: str, b: str) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    return sum(abs(ca.get(k, 0) - cb.get(k, 0)) for k in keys) / max(len(a), 1)


def bootstrap_mean_ci(values: list[float], n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def build_matched_spans(
    seed_path: str,
    *,
    max_families: int = 200,
    seqs_per_family: int = 5,
    seg_min_len: int = 5,
    seg_max_len: int = 8,
    bpe_vocab_size: int = 2000,
    bpe_train_families: int = 1000,
    top_conserved_frac: float = 0.20,
    min_mean_conservation: float = 0.50,
    max_bpe_pieces: int = 4,
    max_composition_l1: float = 1.0,
    seed: int = 0,
):
    """
    Return (spans, meta, bpe_tokenizer).

    `spans` alternates biological / bpe / random matched triplets.
    """
    rng = random.Random(seed)
    examples = collect_examples(
        seed_path,
        max_families=max_families,
        seed=seed,
        seqs_per_family=seqs_per_family,
        seg_min_len=seg_min_len,
        seg_max_len=seg_max_len,
        top_conserved_frac=top_conserved_frac,
    )

    train_seqs: list[str] = []
    for i, fam in enumerate(iter_stockholm_families(seed_path)):
        if i >= bpe_train_families:
            break
        for aln in list(fam.sequences.values())[: max(seqs_per_family, 3)]:
            s = ungapped_sequence(aln)
            if is_standard_protein(s) and 20 <= len(s) <= 1022:
                train_seqs.append(s)
    for ex in examples:
        train_seqs.append(ex.conserved.sequence)
    train_seqs = list(dict.fromkeys(train_seqs))
    bpe = train_protein_bpe(train_seqs, vocab_size=bpe_vocab_size)

    spans: list[SpanItem] = []
    n_skip_cons = n_skip_no_bpe = n_skip_bpe_pieces = n_skip_no_rand = 0
    bpe_piece_hist: dict[int, int] = defaultdict(int)
    length_hist: dict[int, int] = defaultdict(int)

    for i, ex in enumerate(examples):
        bio = ex.conserved
        if bio.mean_conservation < min_mean_conservation:
            n_skip_cons += 1
            continue

        L = bio.length
        group_id = f"{bio.family}:{bio.seq_id}:{bio.start}-{bio.end}:{i}"

        bpe_tok = find_bpe_aligned_span(
            bpe,
            bio.sequence,
            L,
            forbidden=[(bio.start, bio.end)],
            rng=rng,
        )
        if bpe_tok is None:
            n_skip_no_bpe += 1
            continue
        if bpe_tok.n_bpe_tokens > max_bpe_pieces:
            n_skip_bpe_pieces += 1
            continue

        rand = sample_control_window(
            bio.sequence,
            L,
            forbidden=[(bio.start, bio.end), (bpe_tok.start, bpe_tok.end)],
            rng=rng,
            max_tries=400,
            target_segment=bio.segment,
            max_composition_l1=max_composition_l1,
        )
        if rand is None:
            rand = sample_control_window(
                bio.sequence,
                L,
                forbidden=[(bio.start, bio.end), (bpe_tok.start, bpe_tok.end)],
                rng=rng,
                max_tries=400,
                target_segment=bio.segment,
                max_composition_l1=None,
            )
        if rand is None:
            n_skip_no_rand += 1
            continue
        r0, r1 = rand
        rand_seg = bio.sequence[r0:r1]
        if any(c not in AA20 for c in rand_seg):
            n_skip_no_rand += 1
            continue

        bpe_piece_hist[bpe_tok.n_bpe_tokens] += 1
        length_hist[L] += 1

        for kind, start, end, seg, n_tok, comp in (
            ("biological", bio.start, bio.end, bio.segment, 1, 0.0),
            (
                "bpe",
                bpe_tok.start,
                bpe_tok.end,
                bpe_tok.token,
                bpe_tok.n_bpe_tokens,
                composition_l1(bpe_tok.token, bio.segment),
            ),
            (
                "random",
                r0,
                r1,
                rand_seg,
                0,
                composition_l1(rand_seg, bio.segment),
            ),
        ):
            spans.append(
                SpanItem(
                    family=bio.family,
                    seq_id=bio.seq_id,
                    sequence=bio.sequence,
                    start=start,
                    end=end,
                    kind=kind,
                    segment=seg,
                    group_id=group_id,
                    n_bpe_tokens=n_tok,
                    mean_conservation=bio.mean_conservation,
                    composition_l1_to_bio=comp,
                )
            )

    meta = {
        "n_bio_candidates": len(examples),
        "n_matched_triplets": len(spans) // 3,
        "n_skip_low_conservation": n_skip_cons,
        "n_skip_no_bpe": n_skip_no_bpe,
        "n_skip_bpe_pieces": n_skip_bpe_pieces,
        "n_skip_no_rand": n_skip_no_rand,
        "bpe_vocab_size": bpe_vocab_size,
        "bpe_train_sequences": len(train_seqs),
        "bpe_n_tokens_hist": dict(sorted(bpe_piece_hist.items())),
        "length_hist": dict(sorted(length_hist.items())),
        "top_conserved_frac": top_conserved_frac,
        "min_mean_conservation": min_mean_conservation,
        "max_bpe_pieces": max_bpe_pieces,
        "max_composition_l1": max_composition_l1,
        "seg_min_len": seg_min_len,
        "seg_max_len": seg_max_len,
    }
    return spans, meta, bpe


def measure_bpe_cutting(bpe_tokenizer, sequence: str, start: int, end: int) -> dict:
    """How BPE fragments a window [start, end)."""
    units = tokenize_spans(bpe_tokenizer, sequence)
    overlapping = [u for u in units if u.start < end and u.end > start]
    n_pieces = len(overlapping)
    boundaries = {u.end for u in units if start < u.end < end}
    n_internal_cuts = len(boundaries)
    aligned = bool(overlapping) and overlapping[0].start == start and overlapping[-1].end == end
    atomic = n_pieces == 1 and aligned
    # Does a BPE boundary fall exactly at the biological/window edges?
    bpe_starts = {u.start for u in units}
    bpe_ends = {u.end for u in units}
    start_aligned = int(start in bpe_starts)
    end_aligned = int(end in bpe_ends)
    # Tokens that straddle the window edge (cross into/out of the span)
    n_straddle = sum(
        1 for u in overlapping if u.start < start or u.end > end
    )
    return {
        "n_pieces": n_pieces,
        "n_internal_cuts": n_internal_cuts,
        "is_cut": int(n_pieces > 1 or n_internal_cuts > 0),
        "is_aligned": int(aligned),
        "is_atomic": int(atomic),
        "start_boundary_aligned": start_aligned,
        "end_boundary_aligned": end_aligned,
        "both_boundaries_aligned": int(start_aligned and end_aligned),
        "n_straddle": n_straddle,
        "straddle_rate": float(n_straddle / n_pieces) if n_pieces else float("nan"),
    }


def biological_spans_only(spans: list[SpanItem]) -> list[SpanItem]:
    return [s for s in spans if s.kind == "biological"]


def random_spans_only(spans: list[SpanItem]) -> list[SpanItem]:
    return [s for s in spans if s.kind == "random"]


def group_triplets(spans: list[SpanItem]) -> list[dict[str, SpanItem]]:
    by_g: dict[str, dict[str, SpanItem]] = defaultdict(dict)
    for sp in spans:
        by_g[sp.group_id][sp.kind] = sp
    return [v for v in by_g.values() if {"biological", "bpe", "random"} <= set(v.keys())]


def family_wilcoxon_greater(
    rows: list[dict],
    value_key: str,
    kind_a: str,
    kind_b: str,
    seed: int = 0,
) -> dict:
    """Compare kind_a > kind_b on family-level means of value_key."""
    fam: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("kind") not in (kind_a, kind_b):
            continue
        val = r[value_key]
        if val is None or not np.isfinite(val):
            continue
        fam[r["family"]][r["kind"]].append(float(val))

    families = sorted(f for f, d in fam.items() if kind_a in d and kind_b in d)
    xa = [float(np.mean(fam[f][kind_a])) for f in families]
    xb = [float(np.mean(fam[f][kind_b])) for f in families]
    diffs = [u - v for u, v in zip(xa, xb)]
    out = {
        "n_families": len(families),
        f"mean_{kind_a}": float(np.mean(xa)) if xa else None,
        f"mean_{kind_b}": float(np.mean(xb)) if xb else None,
        "mean_diff": float(np.mean(diffs)) if diffs else None,
        "frac_positive": float(np.mean([d > 0 for d in diffs])) if diffs else None,
    }
    if diffs:
        _, lo, hi = bootstrap_mean_ci(diffs, seed=seed)
        out["diff_ci95"] = [lo, hi]
        out["cohens_d"] = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))
        try:
            stat, p = wilcoxon(xa, xb, alternative="greater")
            out["wilcoxon_stat"] = float(stat)
            out["wilcoxon_pvalue"] = float(p)
        except ValueError as e:
            out["wilcoxon_error"] = str(e)
    return out


def add_span_build_args(parser):
    parser.add_argument("--seed-path", default=DEFAULT_SEED)
    parser.add_argument("--max-families", type=int, default=200)
    parser.add_argument("--seqs-per-family", type=int, default=5)
    parser.add_argument("--seg-min-len", type=int, default=5)
    parser.add_argument("--seg-max-len", type=int, default=8)
    parser.add_argument("--top-conserved-frac", type=float, default=0.20)
    parser.add_argument("--min-mean-conservation", type=float, default=0.50)
    parser.add_argument("--max-bpe-pieces", type=int, default=4)
    parser.add_argument("--max-composition-l1", type=float, default=1.0)
    parser.add_argument("--bpe-vocab-size", type=int, default=2000)
    parser.add_argument("--bpe-train-families", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def build_kwargs_from_args(args) -> dict:
    return dict(
        max_families=args.max_families,
        seqs_per_family=args.seqs_per_family,
        seg_min_len=args.seg_min_len,
        seg_max_len=args.seg_max_len,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_train_families=args.bpe_train_families,
        top_conserved_frac=args.top_conserved_frac,
        min_mean_conservation=args.min_mean_conservation,
        max_bpe_pieces=args.max_bpe_pieces,
        max_composition_l1=args.max_composition_l1,
        seed=args.seed,
    )
