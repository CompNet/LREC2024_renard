#!/bin/bash

python xp_ner.py\
       --file_storage='./runs_ner'\
       with\
       seed=0\
       batch_size=4\
       bert_checkpoint="bert-base-cased"
