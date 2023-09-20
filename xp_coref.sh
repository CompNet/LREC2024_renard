#!/bin/bash

if [[ ! -d './litbank' ]]; then
    git clone https://github.com/dbamman/litbank.git
    # checkout a specific commit for reproducibility
    pushd ./litbank
    git checkout '3e50db0ffc033d7ccbb94f4d88f6b99210328ed8'
    popd
fi

python xp_coref.py with litbank_path='./litbank'
