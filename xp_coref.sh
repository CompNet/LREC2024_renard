#!/bin/bash

LITBANK_PATH="${1:-./litbank}"

if [[ ! -d "$LITBANK_PATH" ]]; then
    git clone https://github.com/dbamman/litbank.git "$LITBANK_PATH"
    # checkout a specific commit for reproducibility
    pushd "$LITBANK_PATH"
    git checkout '3e50db0'
    popd
fi

python xp_coref.py --file_storage='./runs_coref' with litbank_path="$LITBANK_PATH"
