"""Dyna-1 ESM-2 exchange predictor (WaymentSteeleLab/Dyna-1)."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np

from utils.alphabet import AA20, blosum_positive_subs, js_bernoulli_vec

DEFAULT_DYNA_CODE = "/home/yuliangyan/Code/Dyna-1"
DEFAULT_DYNA_WEIGHTS = "/mnt/data/models/Dyna-1"


def _is_real_checkpoint(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 10_000:
        return False
    return not path.read_bytes()[:24].startswith(b"version https://git-lfs")


def resolve_dyna_checkpoint(weights_dir: Path, filename: str) -> Path | None:
    candidates = [
        weights_dir / filename,
        weights_dir / "weights" / filename,
        Path(DEFAULT_DYNA_CODE) / "model" / "weights" / filename,
    ]
    for p in candidates:
        if _is_real_checkpoint(p):
            return p
    return None


def _build_dyna_esm2(code_root: Path, ckpt: Path, device):
    """ESM-2 Dyna-1 head only. Avoids importing Dyna-1's ESM-3 stack."""
    import importlib.util

    import torch
    import torch.nn as nn
    from transformers import EsmForTokenClassification

    spec = importlib.util.spec_from_file_location(
        "dyna1_transformer",
        Path(code_root) / "model" / "transformer.py",
    )
    trans_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(trans_mod)
    Transformer = trans_mod.Transformer

    class Dyna1ESM2(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = EsmForTokenClassification.from_pretrained(
                "facebook/esm2_t30_150M_UR50D",
                num_labels=1,
                hidden_dropout_prob=0.0,
                output_hidden_states=True,
                return_dict=True,
                output_attentions=True,
            )
            hidden = backbone.esm.config.hidden_size
            self.model = backbone.esm
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            self.classifier = Transformer(hidden, nheads=8, nlayers=12, dropout=0.1, layer=30)

        def forward(self, tokens, mask):
            out = self.model(input_ids=tokens, attention_mask=mask)
            return self.classifier(out["hidden_states"], torch.logical_not(mask)).squeeze()

    model = Dyna1ESM2()
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_clf = sum(1 for k in state if k.startswith("classifier."))
    print(
        f"  loaded {n_clf} classifier tensors from {ckpt.name} "
        "(ESM-2 backbone is HuggingFace pretrained)",
        flush=True,
    )
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}", flush=True)
    return model


def dyna_prob_adjusted(logits, prior: float = 0.05):
    import torch

    adjustment = math.log(prior / (1.0 - prior))
    return torch.sigmoid(logits - adjustment)


class Dyna1Predictor:
    """p(z_j=1 | x) from Dyna-1. ESM-2 uses sequence; ESM-3 would also take structure."""

    def __init__(self, *, code_root: Path, weights_dir: Path, mode: str, device: str | None = None):
        import torch
        from transformers import AutoTokenizer

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mode = mode
        self._seq_cache: dict[str, np.ndarray] = {}
        ckpt = resolve_dyna_checkpoint(weights_dir, "dyna1-esm2.pt")
        if ckpt is None:
            raise FileNotFoundError(
                f"no real dyna1-esm2.pt under {weights_dir} "
                "(a 134-byte file is a Git LFS pointer, not the weights)"
            )
        print(f"Loading Dyna-1 ESM-2 from {ckpt} on {self.device}", flush=True)
        model = _build_dyna_esm2(code_root, ckpt, self.device)
        model.to(self.device)
        model.eval()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        print(f"  Dyna-1 ready ({mode})", flush=True)

    def predict_seq(self, seq: str) -> np.ndarray:
        hit = self._seq_cache.get(seq)
        if hit is not None:
            return hit
        torch = self.torch
        tokens = self.tokenizer.encode(seq, add_special_tokens=False, return_tensors="pt").to(self.device)
        mask = tokens != 1
        with torch.no_grad():
            logits = self.model(tokens, mask)
            p = dyna_prob_adjusted(logits).detach().float().cpu().numpy().reshape(-1)
        if p.shape[0] != len(seq):
            p = np.resize(p, len(seq))
        self._seq_cache[seq] = p
        return p


def compute_dyna_influence(
    seq: str,
    aa: np.ndarray,
    predictor: Dyna1Predictor,
    sub_cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    q0 = predictor.predict_seq(seq)
    n = len(seq)
    inf = np.zeros((n, n), dtype=np.float64)
    t0 = time.time()
    for i in range(n):
        subs = blosum_positive_subs(int(aa[i]))[:sub_cap]
        if not subs:
            continue
        acc = np.zeros(n, dtype=np.float64)
        for aj in subs:
            mut = seq[:i] + AA20[aj] + seq[i + 1 :]
            q1 = predictor.predict_seq(mut)
            acc += js_bernoulli_vec(q0, q1)
        inf[i] = acc / len(subs)
        if n >= 80 and (i + 1) % 50 == 0:
            print(f"      dyna mutations {i+1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    return q0, inf
