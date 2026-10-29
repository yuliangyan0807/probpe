"""ProBPE geometry, merge scores, and vocabulary learning.

Reusable pieces for scoring weights
    S(u,v) = log f(u,v) + λ_g G + λ_h H + λ_φ Φ + λ_m M + λ_d D
and for training a vocabulary once θ is known.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.alphabet import (
    AA20,
    AA_INDEX,
    BLOSUM62,
    BLOSUM_MAX,
    EPS,
    N_AA,
    aa_idx,
    blosum_positive_subs,
    js_bernoulli_vec,
    sim_blosum,
)
from utils.dyna1 import Dyna1Predictor, compute_dyna_influence

IQR_FLOOR = 1.0  # keep λ=1 on the order of log f; avoid exploding constant scores
LAMBDA_TAU = 1.0 / (3.8**2)  # match radian-scale rotation to Å translation

# Chou–Fasman-like propensities for the toy backbone only.
_HELIX_P = dict(
    zip(
        AA20,
        (1.42, 0.70, 1.01, 1.51, 1.13, 0.57, 1.00, 1.08, 1.16, 1.21, 1.45, 0.67, 0.57, 1.11, 0.98, 0.77, 0.83, 1.06, 1.08, 0.69),
    )
)
_SHEET_P = dict(
    zip(
        AA20,
        (0.83, 1.19, 0.54, 0.37, 1.38, 0.75, 0.87, 1.60, 0.74, 1.30, 1.05, 0.89, 0.55, 1.10, 0.93, 0.75, 1.19, 1.70, 1.37, 1.47),
    )
)


def assign_ss(seq: str) -> str:
    """Local helix / sheet / coil labels from a 5-residue propensity window."""
    n = len(seq)
    out = []
    for i in range(n):
        window = seq[max(0, i - 2) : min(n, i + 3)]
        h = sum(_HELIX_P[c] for c in window) / len(window)
        e = sum(_SHEET_P[c] for c in window) / len(window)
        if h >= 1.05 and h >= e:
            out.append("H")
        elif e >= 1.05:
            out.append("E")
        else:
            out.append("C")
    return "".join(out)


def toy_backbone(seq: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequence-conditioned N/CA/C coordinates (not a PDB substitute).

    Helical stretches pack i,i+3 / i,i+4 contacts; strands are extended;
    coils wander. This lets G, H, Φ vary across candidate tokens when no
    experimental or predicted structure is available.
    """
    n = len(seq)
    ss = assign_ss(seq)
    ca = np.zeros((n, 3), dtype=np.float64)
    frame = np.eye(3)
    origin = np.zeros(3)
    helix_i = strand_i = 0
    for i, lab in enumerate(ss):
        if i == 0:
            ca[i] = origin
        elif lab == "H":
            helix_i += 1
            strand_i = 0
            ang = math.radians(100.0 * helix_i)
            local = np.array([2.3 * math.cos(ang), 2.3 * math.sin(ang), 1.5 * helix_i])
            ca[i] = origin + frame @ local
        elif lab == "E":
            strand_i += 1
            helix_i = 0
            local = np.array([0.9 * (1 if strand_i % 2 == 0 else -1), 0.0, 3.3 * strand_i])
            ca[i] = origin + frame @ local
        else:
            helix_i = strand_i = 0
            step = rng.normal(size=3)
            step /= np.linalg.norm(step) + EPS
            ca[i] = ca[i - 1] + 3.8 * step
            origin = ca[i]
            jitter = rng.normal(size=3)
            jitter /= np.linalg.norm(jitter) + EPS
            z = jitter
            arb = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = np.cross(arb, z)
            x /= np.linalg.norm(x) + EPS
            y = np.cross(z, x)
            frame = np.stack([x, y, z], axis=1)
        if lab != "C" and i > 0 and lab != ss[i - 1]:
            origin = ca[i]
            helix_i = strand_i = 1 if lab != "C" else 0
            z = ca[i] - ca[i - 1]
            z /= np.linalg.norm(z) + EPS
            arb = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = np.cross(arb, z)
            x /= np.linalg.norm(x) + EPS
            y = np.cross(z, x)
            frame = np.stack([x, y, z], axis=1)

    n_xyz = np.zeros_like(ca)
    c_xyz = np.zeros_like(ca)
    for i in range(n):
        prev = ca[i - 1] if i else ca[i] - np.array([1.46, 0.0, 0.0])
        nxt = ca[i + 1] if i + 1 < n else ca[i] + np.array([1.52, 0.0, 0.0])
        tng = nxt - prev
        tng = tng / (np.linalg.norm(tng) + EPS)
        arb = np.array([0.0, 0.0, 1.0]) if abs(tng[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        orth = np.cross(tng, arb)
        orth = orth / (np.linalg.norm(orth) + EPS)
        n_xyz[i] = ca[i] - 1.46 * tng + 0.35 * orth
        c_xyz[i] = ca[i] + 1.52 * tng - 0.35 * orth
    return n_xyz, ca, c_xyz


def residue_frames(n_xyz: np.ndarray, ca: np.ndarray, c_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local frames F_i = (R_i, t_i): Gram-Schmidt of (C−CA, N−CA), origin at CA."""
    v1 = c_xyz - ca
    v2 = n_xyz - ca
    e1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + EPS)
    u2 = v2 - e1 * np.sum(e1 * v2, axis=1, keepdims=True)
    e2 = u2 / (np.linalg.norm(u2, axis=1, keepdims=True) + EPS)
    e3 = np.cross(e1, e2)
    rot = np.stack([e1, e2, e3], axis=-1)  # world = R @ local
    return rot, ca.copy()


def load_backbone_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    data = np.load(path)
    if not {"N", "CA", "C"} <= set(data.files):
        return None
    return data["N"].astype(np.float64), data["CA"].astype(np.float64), data["C"].astype(np.float64)


def relative_poses(rot: np.ndarray, ca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose of residue i+1 in the local frame of residue i, for every peptide bond."""
    n = rot.shape[0]
    rel_r = np.zeros((n - 1, 3, 3), dtype=np.float64)
    rel_t = np.zeros((n - 1, 3), dtype=np.float64)
    for k in range(n - 1):
        rel_r[k] = rot[k].T @ rot[k + 1]
        rel_t[k] = rot[k].T @ (ca[k + 1] - ca[k])
    return rel_r, rel_t


@dataclass
class Protein:
    seq: str
    cluster: str
    n_xyz: np.ndarray
    ca: np.ndarray
    c_xyz: np.ndarray
    rot: np.ndarray
    rel_r: np.ndarray
    rel_t: np.ndarray
    dmat: np.ndarray
    contact: np.ndarray
    aa: np.ndarray
    q_ex: np.ndarray
    influence: np.ndarray
    source: str
    pdb_path: str = ""

    @property
    def L(self) -> int:
        return len(self.seq)


def packing_exchange(contact: np.ndarray, sep_min: int = 3) -> np.ndarray:
    """Static stand-in for Dyna-1 p(z_j=1 | x, C): buried residues exchange less."""
    n = contact.shape[0]
    idx = np.arange(n)
    ncont = np.array(
        [np.sum(contact[i] & (np.abs(idx - i) >= sep_min)) for i in range(n)],
        dtype=np.float64,
    )
    return 1.0 / (1.0 + np.exp(0.35 * (ncont - 8.0)))


def mutated_exchange(q_ex: np.ndarray, contact_i: np.ndarray, i: int, ai: int, a_idx: int) -> np.ndarray:
    """Proxy Dyna-1 response: radical substitutions raise exchange at contacts of i."""
    delta = (BLOSUM62[ai, ai] - BLOSUM62[ai, a_idx]) / BLOSUM_MAX
    q = q_ex.copy()
    q[contact_i] = np.clip(q[contact_i] + 0.2 * max(delta, 0.0) * (1.0 - q[contact_i]), 0.0, 1.0)
    q[i] = float(np.clip(q[i] + 0.1 * max(delta, 0.0), 0.0, 1.0))
    return q


def pair_influence_packing(q_ex: np.ndarray, contact: np.ndarray, aa: np.ndarray, sub_cap: int) -> np.ndarray:
    """J[i, j] from the contact/BLOSUM stand-in (used only if Dyna-1 is off)."""
    n = len(aa)
    inf = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        subs = blosum_positive_subs(int(aa[i]))[:sub_cap]
        if not subs:
            continue
        acc = np.zeros(n, dtype=np.float64)
        for aj in subs:
            q1 = mutated_exchange(q_ex, contact[i], i, int(aa[i]), aj)
            acc += js_bernoulli_vec(q_ex, q1)
        inf[i] = acc / len(subs)
    return inf


def assemble_protein(
    seq: str,
    cluster: str,
    n_xyz: np.ndarray,
    ca: np.ndarray,
    c_xyz: np.ndarray,
    tau: float,
    source: str,
    *,
    pdb_path: str = "",
    predictor: Dyna1Predictor | None = None,
    sub_cap: int = 1,
    dyna_cache_dir: Path | None = None,
) -> Protein:
    rot, _ = residue_frames(n_xyz, ca, c_xyz)
    rel_r, rel_t = relative_poses(rot, ca)
    diff = ca[:, None, :] - ca[None, :, :]
    dmat = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    contact = dmat < tau
    np.fill_diagonal(contact, False)
    aa = aa_idx(seq)
    cache_file = None
    if dyna_cache_dir is not None:
        dyna_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dyna_cache_dir / f"{cluster}.npz"
    q_ex = None
    influence = None
    if cache_file is not None and cache_file.is_file():
        blob = np.load(cache_file)
        if int(blob["q"].shape[0]) == len(seq):
            q_ex = blob["q"].astype(np.float64)
            influence = blob["J"].astype(np.float64)
    if q_ex is None or influence is None:
        if predictor is not None:
            q_ex, influence = compute_dyna_influence(seq, aa, predictor, sub_cap)
        else:
            q_ex = packing_exchange(contact)
            influence = pair_influence_packing(q_ex, contact, aa, sub_cap)
        if cache_file is not None:
            np.savez_compressed(cache_file, q=q_ex, J=influence)
    return Protein(
        seq=seq,
        cluster=cluster,
        n_xyz=n_xyz,
        ca=ca,
        c_xyz=c_xyz,
        rot=rot,
        rel_r=rel_r,
        rel_t=rel_t,
        dmat=dmat,
        contact=contact,
        aa=aa,
        q_ex=q_ex,
        influence=influence,
        source=source,
        pdb_path=pdb_path,
    )


def build_protein(
    seq: str,
    cluster: str,
    rng: np.random.Generator,
    structure_dir: Path | None,
    tau: float,
    sub_cap: int,
    predictor: Dyna1Predictor | None = None,
    dyna_cache_dir: Path | None = None,
) -> Protein:
    coords = None
    source = "toy"
    pdb_path = ""
    if structure_dir is not None:
        coords = load_backbone_npz(structure_dir / f"{cluster}.npz")
        if coords is None:
            coords = load_backbone_npz(structure_dir / f"{cluster}.npy")
    if coords is None:
        n_xyz, ca, c_xyz = toy_backbone(seq, rng)
    else:
        n_xyz, ca, c_xyz = coords
        source = "npz"
        if ca.shape[0] != len(seq):
            n_xyz, ca, c_xyz = toy_backbone(seq, rng)
            source = "toy"
    return assemble_protein(
        seq,
        cluster,
        n_xyz,
        ca,
        c_xyz,
        tau,
        source,
        pdb_path=pdb_path,
        predictor=predictor,
        sub_cap=sub_cap,
        dyna_cache_dir=dyna_cache_dir,
    )


def internal_contact_density(prot: Protein, a: int, b: int, sep_s: int, tau: float) -> float:
    length = b - a + 1
    if length < sep_s + 1:
        return 0.0
    sub = prot.dmat[a : b + 1, a : b + 1]
    ii, jj = np.triu_indices(length, k=sep_s)
    return float(np.mean(sub[ii, jj] < tau))


def long_range_contact_density(prot: Protein, a: int, b: int, delta: int, tau: float) -> float:
    n = prot.L
    idx = np.arange(n)
    inside = (idx >= a) & (idx <= b)
    hits = 0
    for i in range(a, b + 1):
        far = np.abs(idx - i) > delta
        if np.any((~inside) & far & (prot.dmat[i] < tau)):
            hits += 1
    return hits / (b - a + 1)


def relative_pose(prot: Protein, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Pose of residue k+1 in the local frame of residue k."""
    return prot.rel_r[k], prot.rel_t[k]


def _project_so3(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def log_so3_frob2(r: np.ndarray) -> float:
    """||Log(R)||_F^2 = 2 θ^2 for a rotation by angle θ."""
    c = (np.trace(r) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    theta = math.acos(c)
    return 2.0 * theta * theta


def pose_variance(rots: list[np.ndarray], trans: list[np.ndarray]) -> float:
    r_stack = np.stack(rots, axis=0)
    t_stack = np.stack(trans, axis=0)
    r_bar = _project_so3(r_stack.mean(axis=0))
    t_bar = t_stack.mean(axis=0)
    acc = 0.0
    for r, t in zip(rots, trans):
        acc += log_so3_frob2(r_bar.T @ r) + LAMBDA_TAU * float(np.sum((t - t_bar) ** 2))
    return acc / len(rots)


@dataclass
class ScoreCache:
    """Corpus-level statistics reused across θ and fidelities."""

    mu: dict[int, float] = field(default_factory=dict)
    nu: dict[int, float] = field(default_factory=dict)
    sigma2: dict[tuple[int, int], float] = field(default_factory=dict)
    c_dyn: dict[tuple[int, int], float] = field(default_factory=dict)
    kmer_count: Counter = field(default_factory=Counter)
    robust_med: np.ndarray = field(default_factory=lambda: np.zeros(5))
    robust_iqr: np.ndarray = field(default_factory=lambda: np.ones(5))
    max_token: int = 12
    min_variant_support: int = 2
    sep_s: int = 3
    delta: int = 12
    tau: float = 8.0
    min_phi_occ: int = 3
    m_memo: dict = field(default_factory=dict)


def count_kmers(proteins: list[Protein], max_token: int) -> Counter:
    counts: Counter = Counter()
    for p in proteins:
        seq = p.seq
        n = len(seq)
        for k in range(2, max_token + 1):
            for i in range(n - k + 1):
                counts[seq[i : i + k]] += 1
    return counts


def variant_neighborhood(w: str, cache: ScoreCache) -> list[str]:
    if len(w) < 2 or len(w) > cache.max_token:
        return []
    neigh = []
    for i, ch in enumerate(w):
        ai = AA_INDEX[ch]
        for aj in blosum_positive_subs(ai):
            w2 = w[:i] + AA20[aj] + w[i + 1 :]
            if cache.kmer_count[w2] >= cache.min_variant_support:
                neigh.append(w2)
    return neigh


def score_M(w: str, cache: ScoreCache) -> float:
    cached = cache.m_memo.get(w)
    if cached is not None:
        return cached
    neigh = variant_neighborhood(w, cache)
    val = float(np.mean([sim_blosum(w, w2) for w2 in neigh])) if neigh else 0.0
    cache.m_memo[w] = val
    return val


def score_G(occ: list[tuple[Protein, int, int]], cache: ScoreCache) -> float:
    if not occ:
        return 0.0
    length = occ[0][2] - occ[0][1] + 1
    gbar = float(np.mean([internal_contact_density(p, a, b, cache.sep_s, cache.tau) for p, a, b in occ]))
    mu = cache.mu.get(length, gbar)
    return math.log(gbar + EPS) - math.log(mu + EPS)


def score_H(occ: list[tuple[Protein, int, int]], cache: ScoreCache) -> float:
    if not occ:
        return 0.0
    length = occ[0][2] - occ[0][1] + 1
    hbar = float(np.mean([long_range_contact_density(p, a, b, cache.delta, cache.tau) for p, a, b in occ]))
    nu = cache.nu.get(length, hbar)
    return math.log(hbar + EPS) - math.log(nu + EPS)


def score_Phi(pair_occ: list[tuple[Protein, int]], lu: int, lv: int, cache: ScoreCache) -> float:
    if len(pair_occ) < cache.min_phi_occ:
        return 0.0
    rots, trans = [], []
    for prot, k in pair_occ:
        if k < 0 or k >= prot.rel_r.shape[0]:
            continue
        rots.append(prot.rel_r[k])
        trans.append(prot.rel_t[k])
    if len(rots) < cache.min_phi_occ:
        return 0.0
    sig = pose_variance(rots, trans)
    base = cache.sigma2.get((lu, lv), sig)
    return math.log(base + EPS) - math.log(sig + EPS)


def coupling_occurrence(prot: Protein, a: int, k: int, b: int) -> float:
    """C_o(u,v) from the cached J_{i→j} grid (paper: mean over both directions)."""
    lu = k - a + 1
    lv = b - k
    if lu <= 0 or lv <= 0:
        return 0.0
    uv = prot.influence[a : k + 1, k + 1 : b + 1]
    vu = prot.influence[k + 1 : b + 1, a : k + 1]
    if uv.size == 0 or vu.size == 0:
        return 0.0
    return 0.5 * (float(uv.mean()) + float(vu.mean()))


def score_D(
    pair_occ: list[tuple[Protein, int, int, int]],
    lu: int,
    lv: int,
    cache: ScoreCache,
    occ_cap: int,
) -> float:
    if not pair_occ:
        return 0.0
    use = pair_occ[:occ_cap]
    cbar = float(np.mean([coupling_occurrence(p, a, k, b) for p, a, k, b in use]))
    base = cache.c_dyn.get((lu, lv), cbar)
    return math.log(cbar + EPS) - math.log(base + EPS)


def robust_z(raw: np.ndarray, cache: ScoreCache) -> np.ndarray:
    return np.clip((raw - cache.robust_med) / np.maximum(cache.robust_iqr, IQR_FLOOR), -8.0, 8.0)


def _sample_window(proteins: list[Protein], length: int, rng: random.Random) -> tuple[Protein, int, int] | None:
    eligible = [p for p in proteins if p.L >= length]
    if not eligible:
        return None
    p = rng.choice(eligible)
    a = rng.randrange(0, p.L - length + 1)
    return p, a, a + length - 1


def calibrate_cache(
    proteins: list[Protein],
    *,
    max_token: int,
    n_windows: int,
    sep_s: int,
    delta: int,
    tau: float,
    min_variant_support: int,
    min_phi_occ: int,
    sub_cap: int,
    seed: int,
) -> ScoreCache:
    rng = random.Random(seed)
    cache = ScoreCache(
        max_token=max_token,
        min_variant_support=min_variant_support,
        sep_s=sep_s,
        delta=delta,
        tau=tau,
        min_phi_occ=min_phi_occ,
    )
    cache.kmer_count = count_kmers(proteins, max_token)

    for length in range(2, max_token + 1):
        gs, hs = [], []
        for _ in range(n_windows):
            hit = _sample_window(proteins, length, rng)
            if hit is None:
                break
            p, a, b = hit
            gs.append(internal_contact_density(p, a, b, sep_s, tau))
            hs.append(long_range_contact_density(p, a, b, delta, tau))
        cache.mu[length] = float(np.mean(gs)) if gs else 0.0
        cache.nu[length] = float(np.mean(hs)) if hs else 0.0

    for lu in range(1, min(6, max_token) + 1):
        for lv in range(1, min(6, max_token) + 1):
            sigs, cdyn = [], []
            for _ in range(max(32, n_windows // 4)):
                hit = _sample_window(proteins, lu + lv, rng)
                if hit is None:
                    break
                p, a, b = hit
                k = a + lu - 1
                if k + 1 >= p.L:
                    continue
                r, t = relative_pose(p, k)
                sigs.append((r, t))
                cdyn.append(coupling_occurrence(p, a, k, b))
            if len(sigs) >= min_phi_occ:
                cache.sigma2[(lu, lv)] = pose_variance([x[0] for x in sigs], [x[1] for x in sigs])
            if cdyn:
                cache.c_dyn[(lu, lv)] = float(np.mean(cdyn))

    raw_rows = []
    for _ in range(n_windows):
        lu = rng.randint(1, 4)
        lv = rng.randint(1, 4)
        hit = _sample_window(proteins, lu + lv, rng)
        if hit is None:
            continue
        p, a, b = hit
        k = a + lu - 1
        w = p.seq[a : b + 1]
        occ = [(p, a, b)]
        pair_full = [(p, a, k, b)]
        pose_group = []
        for _k in range(cache.min_phi_occ + 2):
            hit2 = _sample_window(proteins, lu + lv, rng)
            if hit2 is None:
                break
            p2, a2, _b2 = hit2
            pose_group.append((p2, a2 + lu - 1))
        raw = np.array(
            [
                score_G(occ, cache),
                score_H(occ, cache),
                score_Phi(pose_group, lu, lv, cache),
                score_M(w, cache),
                score_D(pair_full, lu, lv, cache, occ_cap=4),
            ],
            dtype=np.float64,
        )
        raw_rows.append(raw)
    arr = np.stack(raw_rows, axis=0) if raw_rows else np.zeros((1, 5))
    cache.robust_med = np.median(arr, axis=0)
    q75 = np.percentile(arr, 75, axis=0)
    q25 = np.percentile(arr, 25, axis=0)
    cache.robust_iqr = np.maximum(q75 - q25, IQR_FLOOR)
    _ = sub_cap  # kept so callers can share the Dyna-1 substitution cap
    return cache


@dataclass
class MergeRecord:
    u: str
    v: str
    w: str
    freq: int
    raw: np.ndarray
    z: np.ndarray
    score: float


class ProBPE:
    def __init__(
        self,
        cache: ScoreCache,
        *,
        min_count: int = 2,
        sub_cap: int = 2,
        dyna_occ_cap: int = 6,
        max_token: int = 12,
    ):
        self.cache = cache
        self.min_count = min_count
        self.sub_cap = sub_cap
        self.dyna_occ_cap = dyna_occ_cap
        self.max_token = max_token

    def _pair_map(self, tokenized: list[list[str]], spans: list[list[tuple[int, int]]]):
        occ: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        for pi, toks in enumerate(tokenized):
            for ti in range(len(toks) - 1):
                occ[(toks[ti], toks[ti + 1])].append((pi, ti))
        return occ

    def _score_pair(
        self,
        u: str,
        v: str,
        locations: list[tuple[int, int]],
        proteins: list[Protein],
        spans: list[list[tuple[int, int]]],
        theta: np.ndarray,
    ) -> MergeRecord | None:
        w = u + v
        if len(w) > self.max_token:
            return None
        freq = len(locations)
        token_occ: list[tuple[Protein, int, int]] = []
        pose_occ: list[tuple[Protein, int]] = []
        full_occ: list[tuple[Protein, int, int, int]] = []
        for pi, ti in locations:
            a, mid = spans[pi][ti]
            _, b = spans[pi][ti + 1]
            b = b - 1  # spans are [start, end)
            prot = proteins[pi]
            token_occ.append((prot, a, b))
            k = mid - 1
            pose_occ.append((prot, k))
            full_occ.append((prot, a, k, b))
        lu, lv = len(u), len(v)
        raw = np.array(
            [
                score_G(token_occ, self.cache),
                score_H(token_occ, self.cache),
                score_Phi(pose_occ, lu, lv, self.cache),
                score_M(w, self.cache),
                score_D(full_occ, lu, lv, self.cache, self.dyna_occ_cap),
            ],
            dtype=np.float64,
        )
        z = robust_z(raw, self.cache)
        s = math.log(freq) + float(theta @ z)
        return MergeRecord(u=u, v=v, w=w, freq=freq, raw=raw, z=z, score=s)

    def learn(
        self,
        proteins: list[Protein],
        vocab_size: int,
        theta: np.ndarray,
        log_every: int = 25,
    ) -> tuple[list[MergeRecord], list[str]]:
        tokenized = [list(p.seq) for p in proteins]
        spans = [[(i, i + 1) for i in range(p.L)] for p in proteins]
        vocab = list(AA20)
        rules: list[MergeRecord] = []
        target = max(vocab_size, N_AA)
        t0 = time.time()
        while len(vocab) < target:
            pair_occ = self._pair_map(tokenized, spans)
            best: MergeRecord | None = None
            for (u, v), locs in pair_occ.items():
                if len(locs) < self.min_count:
                    continue
                if len(u) + len(v) > self.max_token:
                    continue
                rec = self._score_pair(u, v, locs, proteins, spans, theta)
                if rec is None:
                    continue
                if best is None or rec.score > best.score:
                    best = rec
            if best is None:
                break
            rules.append(best)
            vocab.append(best.w)
            _apply_merge_inplace(tokenized, spans, best.u, best.v)
            if log_every and len(rules) % log_every == 0:
                print(
                    f"    merge {len(rules):4d}  {best.u}+{best.v}->{best.w}  "
                    f"f={best.freq}  S={best.score:.3f}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        return rules, vocab


def _apply_merge_inplace(
    tokenized: list[list[str]],
    spans: list[list[tuple[int, int]]],
    u: str,
    v: str,
) -> None:
    w = u + v
    for pi, toks in enumerate(tokenized):
        sp = spans[pi]
        new_toks: list[str] = []
        new_sp: list[tuple[int, int]] = []
        k = 0
        while k < len(toks):
            if k + 1 < len(toks) and toks[k] == u and toks[k + 1] == v:
                new_toks.append(w)
                new_sp.append((sp[k][0], sp[k + 1][1]))
                k += 2
            else:
                new_toks.append(toks[k])
                new_sp.append(sp[k])
                k += 1
        tokenized[pi] = new_toks
        spans[pi] = new_sp


def apply_rules(seq: str, rules: list[tuple[str, str]]) -> tuple[list[str], list[bool]]:
    """Replay ordered BPE merges. A rule is activated if it fires at least once."""
    toks = list(seq)
    activated = [False] * len(rules)
    for ri, (u, v) in enumerate(rules):
        w = u + v
        new: list[str] = []
        k = 0
        fired = False
        while k < len(toks):
            if k + 1 < len(toks) and toks[k] == u and toks[k + 1] == v:
                new.append(w)
                fired = True
                k += 2
            else:
                new.append(toks[k])
                k += 1
        toks = new
        activated[ri] = fired
    return toks, activated
