#!/bin/bash

if [[ ! -d './project-dialogism-novel-corpus' ]]; then
    git clone https://github.com/Priya22/project-dialogism-novel-corpus.git
    pushd './project-dialogism-novel-corpus'
    # checkout a specific commit for reproducibility
    git checkout 'b670b9a'
    popd
fi

python xp_characters_extraction.py with PDNC_path='./project-dialogism-novel-corpus'
