# -*- eval: (code-cells-mode); -*-

# %%
from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files={"train": "./data/ner/train.csv", "test": "./data/ner/test.csv"},
)
