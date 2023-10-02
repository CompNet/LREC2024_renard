from typing import Tuple, Union
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from rich import print
from renard.pipeline import Pipeline
from renard.pipeline.ner import BertNamedEntityRecognizer, ner_entities
from renard.pipeline.corefs import BertCoreferenceResolver
from renard.pipeline.characters_extraction import GraphRulesCharactersExtractor
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
from renard_lrec2024.characters_extraction import score_characters_extraction
from renard_lrec2024.network_extraction import (
    load_thg_bio,
    get_thg_characters,
    align_characters,
)
from renard_lrec2024.utils import archive_graph


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore


@ex.config
def config():
    co_occurrences_dist = (10, "tokens")


@ex.automain
def main(_run: Run, co_occurrences_dist: Union[int, Tuple[int, str]]):
    tokens, sentences, gold_bio_tags = load_thg_bio(
        f"./data/network_extraction/TheHungerGames_annotated_no_extended_per_quotesfixed_chaptersfixed.conll"
    )

    # Full extraction pipeline (not using annotations)
    # ------------------------------------------------
    pipeline = Pipeline(
        [
            BertNamedEntityRecognizer(),
            BertCoreferenceResolver(),
            GraphRulesCharactersExtractor(),
            CoOccurrencesGraphExtractor(co_occurrences_dist),
        ]
    )

    out = pipeline(tokens=tokens, sentences=sentences)

    # Gold pipeline using annotations
    # -------------------------------
    gold_pipeline = Pipeline([CoOccurrencesGraphExtractor(co_occurrences_dist)])

    gold_characters = get_thg_characters(tokens)
    gold_out = gold_pipeline(
        tokens=tokens, sentences=sentences, characters=gold_characters
    )

    # Comparison
    # ----------
    archive_graph(_run, out, "thg")
    archive_graph(_run, out, "thg_gold")

    nodes_metrics = score_characters_extraction(
        [character.names for character in gold_out.characters],
        [character.names for character in out.characters],
    )
    print(f"nodes metrics: {nodes_metrics}")
    for k, v in nodes_metrics.items():
        _run.log_scalar(f"nodes_{k}", v)

    # edge recall
    characters_mapping = align_characters(gold_out.characters, out.characters)
    recall_list = []
    for r1, r2 in gold_out.characters_graph.edges:
        c1 = characters_mapping[r1]
        c2 = characters_mapping[r2]
        if (c1, c2) in out.characters_graph.edges:
            recall_list.append(1)
        else:
            recall_list.append(0)
    recall = sum(recall_list) / len(recall_list)
    _run.log_scalar("edge_recall", recall)
    print(f"edges recall: {recall}")

    # edge precision
    precision_list = []
    r_characters_mapping = {
        v: k for k, v in characters_mapping.items() if not v is None
    }
    for c1, c2 in out.characters_graph.edges:
        r1 = r_characters_mapping.get(c1)
        r2 = r_characters_mapping.get(c2)
        if (r1, r2) in gold_out.characters_graph.edges:
            precision_list.append(1)
        else:
            precision_list.append(0)
    precision = sum(precision_list) / len(precision_list)
    _run.log_scalar("edges_precision", precision)
    print(f"edges precision: {precision}")

    f1 = 2 * precision * recall / (precision + recall)
    _run.log_scalar("edges_f1", f1)
    print(f"edges F1: {f1}")
