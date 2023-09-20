import random, argparse, glob
from pathlib import Path
import pandas as pd


random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conivel-dir", type=str)
args = parser.parse_args()


owto_dataset_dir = Path(args.conivel_dir) / "conivel/datas/dekker/dataset"
conll_files = sorted(glob.glob(f"{owto_dataset_dir}/*.conll"))
random.shuffle(conll_files)
train_set = conll_files[: int(0.8 * len(conll_files))]
test_set = conll_files[int(0.8 * len(conll_files)) :]


tag_to_id = {"O": 0, "B-PER": 1, "I-PER": 2}


def load_into_df(conll_file: str) -> pd.DataFrame:
    tokens = []
    tags = []

    with open(conll_file) as f:

        for i, line in enumerate(f):

            token, tag = line.strip().split(" ")

            tokens.append(token)
            tags.append(tag_to_id.get(tag, tag_to_id["O"]))

    # parse into sentences
    sents = []
    sent = {"tokens": [], "labels": []}

    for i, (token, tag) in enumerate(zip(tokens, tags)):

        fixed_token = '"' if token in {"``", "''"} else token
        fixed_token = "'" if token == "`" else fixed_token
        next_token = tokens[i + 1] if i < len(tokens) - 1 else None

        sent["tokens"].append(fixed_token)
        sent["labels"].append(tag)

        # quote ends next token : skip this token
        # this avoids problem with cases where we have punctuation
        # at the end of a quote (otherwise, the end of the quote
        # would be part of the next sentence)
        if next_token == "''":
            continue

        # sentence end
        if token in ["''", ".", "?", "!"]:
            sents.append(sent)
            sent = {"tokens": [], "labels": []}

    return pd.DataFrame(sents)


train_set = pd.concat([load_into_df(f) for f in train_set], ignore_index=True)
test_set = pd.concat([load_into_df(f) for f in test_set], ignore_index=True)

train_set.to_csv(Path("__file__").parents[0] / "train.csv")
test_set.to_csv(Path("__file__").parents[0] / "test.csv")
