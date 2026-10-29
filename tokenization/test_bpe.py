"""Unit tests for the production vanilla protein BPE tokenizer."""

from __future__ import annotations

import json

import pytest
from tokenizers import Tokenizer

from bpe import (
    BPETrainingConfig,
    SPECIAL_TOKENS,
    SequenceStats,
    prepare_sequences,
    sequence_from_afdb_pdb,
    train_bpe,
    train_from_file,
)


def test_training_reaches_requested_vocab_and_round_trips() -> None:
    sequences = ["ACDEFGHIKLMNPQRSTVWY" * 4, "ACDACDACDACD", "LMNLMNLMN"]
    config = BPETrainingConfig(vocab_size=50, min_frequency=1)
    tokenizer = train_bpe(sequences, config, show_progress=False)

    assert tokenizer.get_vocab_size() == 50
    assert all(token in tokenizer.get_vocab() for token in SPECIAL_TOKENS)

    sequence = "ACDACDLMN"
    encoding = tokenizer.encode(sequence)
    assert encoding.tokens[0] == "<cls>"
    assert encoding.tokens[-1] == "<eos>"
    assert tokenizer.decode(encoding.ids) == sequence

    content = tokenizer.encode(sequence, add_special_tokens=False)
    assert "".join(content.tokens) == sequence
    assert content.offsets[0][0] == 0
    assert content.offsets[-1][1] == len(sequence)


def test_invalid_residue_policies() -> None:
    replace_config = BPETrainingConfig(
        vocab_size=40,
        invalid_residue_policy="replace",
    )
    stats = SequenceStats()
    assert list(prepare_sequences(["AC?D"], replace_config, stats)) == ["ACXD"]
    assert stats.sequences_with_invalid_residues == 1

    skip_config = BPETrainingConfig(
        vocab_size=40,
        invalid_residue_policy="skip",
    )
    assert list(prepare_sequences(["AC?D", "ACDE"], skip_config)) == ["ACDE"]

    error_config = BPETrainingConfig(vocab_size=40)
    with pytest.raises(ValueError, match="invalid residue symbols"):
        list(prepare_sequences(["AC?D"], error_config))


def test_cli_training_helpers_stream_fasta_and_write_metadata(tmp_path) -> None:
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(
        ">protein_1\nACDEFGHIKL\nMNPQRSTVWY\n>protein_2\nACDACDACD\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "tokenizers"

    paths = train_from_file(
        fasta,
        output_dir,
        [40],
        min_frequency=1,
        show_progress=False,
    )

    tokenizer = Tokenizer.from_file(str(paths[0]))
    metadata = json.loads(paths[0].with_suffix(".meta.json").read_text())
    assert tokenizer.get_vocab_size() == 40
    assert metadata["actual_vocab_size"] == 40
    assert metadata["sequence_stats"]["sequences_used"] == 2
    assert metadata["sequence_stats"]["residues_used"] == 29
    sequence = "ACDEFGHIKL"
    assert tokenizer.decode(tokenizer.encode(sequence).ids) == sequence


def test_sequence_from_afdb_pdb(tmp_path) -> None:
    pdb = tmp_path / "AF-P12345-F1-model_v6.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 90.00           C  \n"
        "ATOM      2  CA  CYS A   2      11.000  10.000  10.000  1.00 90.00           C  \n"
        "ATOM      3  CA  ASP A   3      12.000  10.000  10.000  1.00 90.00           C  \n",
        encoding="utf-8",
    )
    assert sequence_from_afdb_pdb(pdb) == "ACD"
