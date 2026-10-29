#!/usr/bin/env bash

set -euo pipefail

# Resolve the repository path from this script so the command works from any
# current working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

conda run -n mgd python tokenization/bpe.py \
    --input /mnt/data/dataset/afdb/swissprot \
    --format afdb \
    --output-dir tokenization/tokenizers/vanilla_bpe \
    --vocab-size 100 500 2000 10000 \
    --min-frequency 2
