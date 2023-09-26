from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Set
import pandas as pd
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from rich import print
from tqdm import tqdm
from renard.pipeline import Pipeline
from renard.pipeline.corefs import BertCoreferenceResolver
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.characters_extraction import GraphRulesCharactersExtractor
from renard_lrec2024.characters_extraction import (
    score_characters_extraction,
    PDNCPerfectNamedEntityRecognizer,
)


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
    use_coref: bool = False


@ex.automain
def main(_run: Run, PDNC_path: str, use_coref: bool):

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
            if row["Main Name"] in ("_group", "_unknowable"):
                continue
            book_characters.append({row["Main Name"], *eval(row["Aliases"])})

        with open(d / "novel_text.txt") as f:
            text = f.read()

        PDNC.append(PDNCBook(d.name, text, book_characters))

    # Inference
    # ---------
    # the PDNC dataset only annotates speaker: therefore, we cant use
    # a named entity recognizer to extract entities since that would
    # extract extra unannotated characters. Instead, we consider all
    # mentions of given characters, simulating perfect NER.
    steps = [NLTKTokenizer(), PDNCPerfectNamedEntityRecognizer()]
    if use_coref:
        steps.append(BertCoreferenceResolver())
    steps.append(GraphRulesCharactersExtractor(link_corefs_mentions=False))
    pipeline = Pipeline(steps, warn=False, progress_report=None)

    predicted_characters = []
    for book in tqdm(PDNC):
        # NOTE: pass in refs at run time for PDNCPerfectNamedEntityRecognizer
        out = pipeline(book.text, refs=book.characters)
        assert not out.characters is None

        # extract predictions
        preds = [set(char.names) for char in out.characters]
        predicted_characters.append(preds)

        # restrict labels to characters found in the text, since some
        # of them aren't present.
        all_extracted_names = {" ".join(entity.tokens) for entity in out.entities}
        # 1. remove names not extracted by PDNCPerfectNamedEntityRecognizer
        book.characters = [
            {name for name in names if name in all_extracted_names}
            for names in book.characters
        ]
        # 2. the previous operation might have left some empty name
        #    sets: delete them
        book.characters = [names for names in book.characters if not len(names) == 0]

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
