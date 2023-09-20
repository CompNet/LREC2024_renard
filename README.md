# Renard at LREC 2024

This repository contains the code supporting the LREC 2024 article "Renard: A Modular Pipeline for Extracting Character Networks from Narrative Texts" (for the Renard repository, see [here](https://github.com/CompNet/Renard)).


# Reproducing results

First, install the dependencies with `pip install -r requirements`. 

Shell scripts are available to reproduce all of the article's experiments:

| Experiment                         | Script                        |
|------------------------------------|-------------------------------|
| NER performance                    | `xp_ner.sh`                   |
| Speaker Attribution performance    | `xp_speaker_attribution.sh`   |
| Coreference Resolution performance | `xp_coref.sh`                 |
| Characters Extraction performance  | `xp_characters_extraction.sh` |
