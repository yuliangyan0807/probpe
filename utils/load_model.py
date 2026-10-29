import os

# huggingface.co is often unreachable in CN; use mirror before importing hub clients
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# DEFAULT_ESM_MODEL = "facebook/esm2_t36_3B_UR50D"
DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"


def load_esm_model(model_name=DEFAULT_ESM_MODEL, dtype=None):
    """Load a frozen ESM-2 Masked LM and its tokenizer."""
    if dtype is None:
        # bf16 can produce NaN logits on some ESM-2 scales/GPUs; prefer fp32.
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_esm_model()
    print(model)
