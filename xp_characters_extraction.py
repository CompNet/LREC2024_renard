from pathlib import Path
from collections import defaultdict
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


@ex.config
def config():
    PDNC_path: str


@ex.automain
def main(_run: Run, PDNC_path: str):

    # Load PDNC dataset
    # -----------------
    PDNC_texts = []
    PDNC_characters = []

    for d in Path(PDNC_path).iterdir():

        if not d.is_dir():
            continue

        characters_infos = pd.read_csv(d / "character_info.csv")
        for _, row in characters_infos.iterrows():
            PDNC_characters.append({row["Main Name"], *row["Aliases"]})

        with open(d / "novel_text.txt") as f:
            PDNC_texts.append(f.read())

    # Inference
    # ---------
    pipeline = Pipeline(
        # NOTE: we have to perform at least tokenization and NER as
        # inputs to characters extraction
        [NLTKTokenizer(), BertNamedEntityRecognizer(), GraphRulesCharactersExtractor()]
    )
    predicted_characters = []
    for text in PDNC_texts:
        out = pipeline(text)
        predicted_characters.append([char.names for char in out.characters])

    # Scoring
    # -------
    # we perform scoring as in Vala et. al, 2015
    metrics_dict = defaultdict(list)

    for refs, preds in zip(PDNC_characters, predicted_characters):
        metrics = score_characters_extraction(refs, preds)
        for k, v in metrics.items():
            metrics_dict[k].append(v)

    # mean of all metrics across books
    metrics_dict = {k: sum(v) / len(v) for k, v in metrics_dict.items()}

    for k, v in metrics_dict:
        _run.log_scalar(k, v)
    print(metrics_dict)
