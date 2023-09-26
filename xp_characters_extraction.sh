#!/bin/bash

PDNC_PATH="${1:-./project-dialogism-novel-corpus}"

if [[ ! -d "$PDNC_PATH" ]]; then
    git clone https://github.com/Priya22/project-dialogism-novel-corpus.git "$PDNC_PATH"
    pushd "$PDNC_PATH"
    # checkout a specific commit for reproducibility
    git checkout 'b670b9a'
    popd
fi

python xp_characters_extraction.py\
       --file_storage='./runs_characters_extraction'\
       with\
       seed=0\
       PDNC_path="$PDNC_PATH"\
       use_coref=False

python xp_characters_extraction.py\
       --file_storage='./runs_characters_extraction'\
       with\
       seed=0\
       PDNC_path="$PDNC_PATH"\
       use_coref=True
