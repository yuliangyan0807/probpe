"""Train a vanilla BPE tokenizer for protein sequences.

The implementation intentionally follows standard LLM BPE: training starts
from a character alphabet and repeatedly merges the most frequent adjacent
pair.  Protein-specific behavior is limited to the input alphabet and sequence
validation; no structural or evolutionary score is used here.

Example
-------
Train the four vocabularies used in the tokenizer-scaling experiments::

    python tokenization/bpe.py \
        --input swissprot.fasta.gz \
        --output-dir tokenizers/vanilla_bpe \
        --vocab-size 100 500 2000 10000
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

# This is the residue alphabet used by the original ESM-2 tokenizer.  Period
# and hyphen are retained for vocabulary compatibility, although raw UniProt
# sequences normally do not contain alignment symbols.
ESM2_RESIDUE_TOKENS: tuple[str, ...] = tuple("LAGVSERTIDPKQNFYMHWCXBUZO.-")

# Keep the ESM-2 spellings so a future ESM-2 model wrapper can obtain every
# required token id directly from the tokenizer.  <null_1> is a reserved token
# in the original 33-token ESM-2 vocabulary and is never emitted from a raw
# protein sequence.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "<null_1>",
    "<mask>",
)

UNK_TOKEN = "<unk>"
VALID_RESIDUES = frozenset(ESM2_RESIDUE_TOKENS)

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class BPETrainingConfig:
    """Configuration recorded next to every trained tokenizer."""

    vocab_size: int
    min_frequency: int = 2
    max_token_length: int | None = None
    min_sequence_length: int = 1
    max_sequence_length: int | None = None
    invalid_residue_policy: str = "error"
    max_sequences: int | None = None

    def validate(self) -> None:
        minimum_vocab = len(ESM2_RESIDUE_TOKENS) + len(SPECIAL_TOKENS)
        if self.vocab_size < minimum_vocab:
            raise ValueError(
                f"vocab_size must be at least {minimum_vocab} to contain the "
                "ESM-2 residue alphabet and special tokens"
            )
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        if self.max_token_length is not None and self.max_token_length < 1:
            raise ValueError("max_token_length must be positive when provided")
        if self.min_sequence_length < 1:
            raise ValueError("min_sequence_length must be positive")
        if (
            self.max_sequence_length is not None
            and self.max_sequence_length < self.min_sequence_length
        ):
            raise ValueError(
                "max_sequence_length must be no smaller than min_sequence_length"
            )
        if self.invalid_residue_policy not in {"error", "replace", "skip"}:
            raise ValueError(
                "invalid_residue_policy must be one of: error, replace, skip"
            )
        if self.max_sequences is not None and self.max_sequences < 1:
            raise ValueError("max_sequences must be positive when provided")


@dataclass
class SequenceStats:
    """Input statistics collected while a tokenizer is being trained."""

    records_read: int = 0
    sequences_used: int = 0
    sequences_too_short: int = 0
    sequences_too_long: int = 0
    sequences_with_invalid_residues: int = 0
    residues_used: int = 0


def _open_text(path: Path) -> TextIO:
    """Open a plain-text or gzip-compressed sequence file."""

    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="rt", encoding="utf-8")


def _detect_format(path: Path) -> str:
    """Infer FASTA versus one-sequence-per-line text from the file name."""

    if path.is_dir():
        return "afdb"
    name = path.name.removesuffix(".gz").lower()
    if name.endswith((".fa", ".faa", ".fasta", ".fas")):
        return "fasta"
    return "text"


def sequence_from_afdb_pdb(path: str | Path) -> str:
    """Extract a one-letter sequence from an AlphaFold DB PDB file.

    One CA atom is used per residue.  AlphaFold DB models contain a complete
    single-chain prediction, so this avoids loading coordinates that vanilla
    BPE does not need while preserving the sequence associated with the
    structure for later structure-guided tokenizers.
    """

    pdb_path = Path(path)
    opener = gzip.open if pdb_path.suffix == ".gz" else open
    residues: list[str] = []
    seen_positions: set[tuple[str, str, str]] = set()

    with opener(pdb_path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            # Ignore secondary alternate locations for the same residue.
            if line[16:17] not in {" ", "A"}:
                continue
            residue = AA3_TO_1.get(line[17:20].strip())
            if residue is None:
                continue
            position = (line[21:22], line[22:26], line[26:27])
            if position in seen_positions:
                continue
            seen_positions.add(position)
            residues.append(residue)

    if not residues:
        raise ValueError(f"no standard amino-acid residues found in {pdb_path}")
    return "".join(residues)


def iter_afdb_sequences(path: str | Path) -> Iterator[str]:
    """Yield sequences from all AlphaFold DB PDB files in a directory."""

    afdb_dir = Path(path)
    if not afdb_dir.is_dir():
        raise NotADirectoryError(f"AlphaFold DB input is not a directory: {afdb_dir}")

    # Sorting makes BPE tie-breaking reproducible across file-system layouts.
    names = sorted(
        entry.name
        for entry in os.scandir(afdb_dir)
        if entry.is_file()
        and entry.name.startswith("AF-")
        and entry.name.endswith((".pdb", ".pdb.gz"))
    )
    if not names:
        raise FileNotFoundError(f"no AF-*.pdb or AF-*.pdb.gz files in {afdb_dir}")

    for name in names:
        yield sequence_from_afdb_pdb(afdb_dir / name)


def iter_raw_sequences(path: str | Path, file_format: str = "auto") -> Iterator[str]:
    """Yield sequences from AFDB PDBs, FASTA, or line-oriented text.

    Records are streamed and never loaded into memory as a complete corpus.
    """

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"sequence file does not exist: {input_path}")

    resolved_format = _detect_format(input_path) if file_format == "auto" else file_format
    if resolved_format not in {"afdb", "fasta", "text"}:
        raise ValueError("file_format must be one of: auto, afdb, fasta, text")

    if resolved_format == "afdb":
        yield from iter_afdb_sequences(input_path)
        return

    with _open_text(input_path) as handle:
        if resolved_format == "text":
            for line in handle:
                sequence = "".join(line.split()).upper()
                if sequence:
                    yield sequence
            return

        chunks: list[str] = []
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    yield "".join(chunks).upper()
                    chunks.clear()
                continue
            if not chunks and line_number == 1:
                raise ValueError(
                    f"FASTA input must start with a header line: {input_path}"
                )
            chunks.append("".join(line.split()))
        if chunks:
            yield "".join(chunks).upper()


def prepare_sequences(
    sequences: Iterable[str],
    config: BPETrainingConfig,
    stats: SequenceStats | None = None,
) -> Iterator[str]:
    """Validate, filter, and stream sequences into the BPE trainer.

    Long sequences are skipped rather than truncated because tokenizer
    learning is not constrained by the language model context window.  This
    prevents an arbitrary crop from changing pair-frequency statistics.
    """

    config.validate()
    current_stats = stats if stats is not None else SequenceStats()

    for raw_sequence in sequences:
        current_stats.records_read += 1
        sequence = "".join(raw_sequence.split()).upper()

        if len(sequence) < config.min_sequence_length:
            current_stats.sequences_too_short += 1
            continue
        if (
            config.max_sequence_length is not None
            and len(sequence) > config.max_sequence_length
        ):
            current_stats.sequences_too_long += 1
            continue

        invalid = sorted(set(sequence) - VALID_RESIDUES)
        if invalid:
            current_stats.sequences_with_invalid_residues += 1
            if config.invalid_residue_policy == "error":
                symbols = "".join(invalid)
                raise ValueError(
                    f"sequence {current_stats.records_read} contains invalid "
                    f"residue symbols: {symbols!r}"
                )
            if config.invalid_residue_policy == "skip":
                continue
            sequence = "".join(
                residue if residue in VALID_RESIDUES else "X"
                for residue in sequence
            )

        current_stats.sequences_used += 1
        current_stats.residues_used += len(sequence)
        yield sequence

        if (
            config.max_sequences is not None
            and current_stats.sequences_used >= config.max_sequences
        ):
            return


def train_bpe(
    sequences: Iterable[str],
    config: BPETrainingConfig,
    *,
    show_progress: bool = True,
    stats: SequenceStats | None = None,
) -> Tokenizer:
    """Train and return a vanilla protein BPE tokenizer.

    ``vocab_size`` is the final model vocabulary size, including special
    tokens.  The function does not install a character-splitting pre-tokenizer:
    the Hugging Face BPE model therefore learns merges over each complete
    protein sequence, exactly as required for amino-acid substrings.
    """

    config.validate()
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    trainer_kwargs: dict[str, object] = {
        "vocab_size": config.vocab_size,
        "min_frequency": config.min_frequency,
        "special_tokens": list(SPECIAL_TOKENS),
        "initial_alphabet": list(ESM2_RESIDUE_TOKENS),
        "show_progress": show_progress,
    }
    if config.max_token_length is not None:
        trainer_kwargs["max_token_length"] = config.max_token_length

    trainer = BpeTrainer(**trainer_kwargs)
    tokenizer.train_from_iterator(
        prepare_sequences(sequences, config, stats),
        trainer=trainer,
    )
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != config.vocab_size:
        raise ValueError(
            f"BPE training produced {actual_vocab_size} tokens instead of the "
            f"requested {config.vocab_size}. The corpus may be too small, "
            "min_frequency may be too high, or max_token_length may be too low."
        )

    # Protein BPE pieces must be concatenated without whitespace during
    # decoding.  The post-processor makes the serialized tokenizer directly
    # usable by ESM-2-style pretraining: encode() adds <cls> and <eos>, while
    # encode(..., add_special_tokens=False) remains available for raw spans.
    tokenizer.decoder = Fuse()
    tokenizer.post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        special_tokens=[
            ("<cls>", tokenizer.token_to_id("<cls>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    return tokenizer


def truncate_bpe(tokenizer: Tokenizer, vocab_size: int) -> Tokenizer:
    """Create a smaller BPE tokenizer from a prefix of learned merge rules.

    BPE is hierarchical: the first ``k`` merges of a larger tokenizer are
    exactly the merges needed by its smaller-vocabulary counterpart.  Training
    the largest vocabulary once therefore avoids parsing the 28 GB Swiss-Prot
    structure directory once per requested vocabulary size.
    """

    minimum_vocab = len(ESM2_RESIDUE_TOKENS) + len(SPECIAL_TOKENS)
    if not minimum_vocab <= vocab_size <= tokenizer.get_vocab_size():
        raise ValueError(
            f"vocab_size must be between {minimum_vocab} and "
            f"{tokenizer.get_vocab_size()}"
        )

    data = json.loads(tokenizer.to_str())
    full_vocab = data["model"]["vocab"]
    kept_vocab = {
        token: token_id for token, token_id in full_vocab.items() if token_id < vocab_size
    }
    kept_tokens = set(kept_vocab)
    kept_merges = []
    for merge in data["model"]["merges"]:
        left, right = merge if isinstance(merge, list) else merge.split(" ")
        if left + right in kept_tokens:
            kept_merges.append(merge)

    data["model"]["vocab"] = kept_vocab
    data["model"]["merges"] = kept_merges
    truncated = Tokenizer.from_str(json.dumps(data))
    if truncated.get_vocab_size() != vocab_size:
        raise RuntimeError("failed to construct the requested BPE vocabulary prefix")
    return truncated


def save_tokenizer(
    tokenizer: Tokenizer,
    output_path: str | Path,
    config: BPETrainingConfig,
    stats: SequenceStats,
    *,
    source_path: str | Path | None = None,
) -> None:
    """Save tokenizer JSON and a human-readable metadata sidecar."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))

    metadata = {
        "tokenizer_type": "vanilla_bpe",
        "source_path": str(source_path) if source_path is not None else None,
        "requested_vocab_size": config.vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "config": asdict(config),
        "sequence_stats": asdict(stats),
        "special_tokens": list(SPECIAL_TOKENS),
        "initial_alphabet": list(ESM2_RESIDUE_TOKENS),
    }
    metadata_path = path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_from_file(
    input_path: str | Path,
    output_dir: str | Path,
    vocab_sizes: Sequence[int],
    *,
    min_frequency: int = 2,
    max_token_length: int | None = None,
    min_sequence_length: int = 1,
    max_sequence_length: int | None = None,
    invalid_residue_policy: str = "error",
    max_sequences: int | None = None,
    file_format: str = "auto",
    show_progress: bool = True,
) -> list[Path]:
    """Train one tokenizer for each requested vocabulary size."""

    if not vocab_sizes:
        raise ValueError("at least one vocabulary size is required")
    if len(set(vocab_sizes)) != len(vocab_sizes):
        raise ValueError("vocabulary sizes must not contain duplicates")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    maximum_vocab = max(vocab_sizes)
    maximum_config = BPETrainingConfig(
        vocab_size=maximum_vocab,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
        min_sequence_length=min_sequence_length,
        max_sequence_length=max_sequence_length,
        invalid_residue_policy=invalid_residue_policy,
        max_sequences=max_sequences,
    )
    stats = SequenceStats()
    raw_sequences = iter_raw_sequences(input_path, file_format=file_format)
    maximum_tokenizer = train_bpe(
        raw_sequences,
        maximum_config,
        show_progress=show_progress,
        stats=stats,
    )

    # Smaller vocabularies reuse prefixes of the same learned merge hierarchy.
    # This both guarantees nesting and scans the AFDB corpus only once.
    for vocab_size in vocab_sizes:
        config = BPETrainingConfig(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            max_token_length=max_token_length,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            invalid_residue_policy=invalid_residue_policy,
            max_sequences=max_sequences,
        )
        tokenizer = (
            maximum_tokenizer
            if vocab_size == maximum_vocab
            else truncate_bpe(maximum_tokenizer, vocab_size)
        )
        output_path = destination / f"bpe_{vocab_size}.json"
        save_tokenizer(
            tokenizer,
            output_path,
            config,
            stats,
            source_path=input_path,
        )
        saved_paths.append(output_path)

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train vanilla BPE tokenizers on protein sequences."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Swiss-Prot AFDB directory, FASTA(.gz), or line-oriented text input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for tokenizer JSON and metadata files.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        nargs="+",
        required=True,
        help="Final vocabulary sizes, including special tokens.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum corpus frequency required for a merge (default: 2).",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=None,
        help="Optional maximum residues per learned token; default is unlimited.",
    )
    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=1,
        help="Skip sequences shorter than this many residues (default: 1).",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Skip longer sequences; default keeps full proteins.",
    )
    parser.add_argument(
        "--invalid-residue-policy",
        choices=("error", "replace", "skip"),
        default="error",
        help="Handle symbols outside the ESM-2 residue alphabet (default: error).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional number of accepted sequences, useful for smoke tests.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "afdb", "fasta", "text"),
        default="auto",
        help="Input format; auto detects an AFDB directory or file extension.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tokenizer training progress bar.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = train_from_file(
        input_path=args.input,
        output_dir=args.output_dir,
        vocab_sizes=args.vocab_size,
        min_frequency=args.min_frequency,
        max_token_length=args.max_token_length,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
        invalid_residue_policy=args.invalid_residue_policy,
        max_sequences=args.max_sequences,
        file_format=args.format,
        show_progress=not args.no_progress,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
