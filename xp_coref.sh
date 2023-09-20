#!/bin/bash

if [[ ! -d './litbank' ]]; then
    git clone https://github.com/dbamman/litbank.git
fi

python xp_coref.py with litbank_path='./litbank'
