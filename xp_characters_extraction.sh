#!/bin/bash

PDNC_PATH="./project-dialogism-novel-corpus"
if [[ ! -z "$1" ]];then
    PDNC_PATH="$1"
fi

if [[ ! -d "$PDNC_PATH" ]]; then
    git clone https://github.com/Priya22/project-dialogism-novel-corpus.git
    pushd "$PDNC_PATH"
    # checkout a specific commit for reproducibility
    git checkout 'b670b9a'
    popd
fi

python xp_characters_extraction.py with PDNC_path="$PDNC_PATH"
