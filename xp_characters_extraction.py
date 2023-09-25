from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Set
import pandas as pd
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from rich import print
from renard.pipeline import Pipeline
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.ner import BertNamedEntityRecognizer
from renard.pipeline.characters_extraction import GraphRulesCharactersExtractor
from characters_extraction import score_characters_extraction


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore


@dataclass
class PDNCBook:
    title: str
    text: str
    characters: List[Set[str]]


@ex.config
def config():
    PDNC_path: str


@ex.automain
def main(_run: Run, PDNC_path: str):

    # Load PDNC dataset
    # -----------------
    PDNC: List[PDNCBook] = []

    data_path = Path(PDNC_path) / "data"
    for d in data_path.iterdir():

        if not d.is_dir():
            continue

        characters_infos = pd.read_csv(d / "character_info.csv")
        book_characters = []
        for _, row in characters_infos.iterrows():
            book_characters.append({row["Main Name"], *eval(row["Aliases"])})

        with open(d / "novel_text.txt") as f:
            text = f.read()

        PDNC.append(PDNCBook(d.name, text, book_characters))

    # TODO: dev
    # keep only the shortest book, for performance reasons
    PDNC = [min(PDNC, key=lambda b: len(b.text))]

    # Inference
    # ---------
    pipeline = Pipeline(
        # NOTE: we have to perform at least tokenization and NER as
        # inputs to characters extraction
        [NLTKTokenizer(), BertNamedEntityRecognizer(), GraphRulesCharactersExtractor()]
    )
    predicted_characters = []
    for book in PDNC:
        out = pipeline(book.text)
        predicted_characters.append([char.names for char in out.characters])

    # Scoring
    # -------
    # we perform scoring as in Vala et. al, 2015
    metrics_dict = defaultdict(list)
    for book, preds in zip(PDNC, predicted_characters):
        metrics = score_characters_extraction(book.characters, preds)
        for k, v in metrics.items():
            metrics_dict[k].append(v)
            _run.log_scalar(f"{book.title}.{k}", v)
            print(f"{book.title}.{k}: {v}")

    # mean of all metrics across books
    metrics_dict = {k: sum(v) / len(v) for k, v in metrics_dict.items()}

    for k, v in metrics_dict.items():
        _run.log_scalar(k, v)
    print(metrics_dict)
