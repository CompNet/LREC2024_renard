# Renard at LREC 2024

This repository contains the code supporting the LREC 2024 article "Renard: A Modular Pipeline for Extracting Character Networks from Narrative Texts" (for the Renard repository, see [here](https://github.com/CompNet/Renard)).


# Reproducing results

First, install the dependencies with `pip install -r requirements` (or `poetry install`). 

Shell scripts are available to reproduce all of the article's experiments:

| Experiment             | Script                        | Output directory             |
|------------------------|-------------------------------|------------------------------|
| NER                    | `xp_ner.sh`                   | `runs_ner`                   |
| Speaker Attribution    | `xp_speaker_attribution.sh`   | `runs_speaker_attribution`   |
| Coreference Resolution | `xp_coref.sh`                 | `runs_coref`                 |
| Characters Extraction  | `xp_characters_extraction.sh` | `runs_characters_extraction` |
| Network Extraction     | `xp_network_extraction.sh`    | `runs_network_extraction`    |
