from typing import Tuple, Union
from more_itertools.recipes import flatten
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from rich import print
from renard.pipeline import Pipeline
from renard.pipeline.ner import BertNamedEntityRecognizer
from renard.pipeline.corefs import BertCoreferenceResolver
from renard.pipeline.characters_extraction import GraphRulesCharactersExtractor
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
from renard_lrec2024.characters_extraction import score_characters_extraction
from renard_lrec2024.network_extraction import (
    load_thg_bio,
    get_thg_characters,
    align_characters,
    score_network_extraction_edges,
    shared_layout,
)
from renard_lrec2024.utils import archive_graph, archive_pipeline_state


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore


@ex.config
def config():
    co_occurrences_dist = (1, "sentences")


@ex.automain
def main(_run: Run, co_occurrences_dist: Union[int, Tuple[int, str]]):
    tokens, sentences, _ = load_thg_bio(
        f"./data/network_extraction/TheHungerGames_annotated_no_extended_per_quotesfixed_chaptersfixed.conll"
    )

    # TODO: dev
    sentences = sentences[:1000]
    tokens = list(flatten(sentences))

    # Full extraction pipeline
    # ------------------------
    full_pipeline = Pipeline(
        [
            BertNamedEntityRecognizer(),
            BertCoreferenceResolver(),
            GraphRulesCharactersExtractor(),
            CoOccurrencesGraphExtractor(co_occurrences_dist),
        ]
    )

    full_out = full_pipeline(tokens=tokens, sentences=sentences)
    archive_pipeline_state(_run, full_out, "full_pipeline_state")

    # -corefs extraction pipeline
    # ---------------------------
    no_corefs_pipeline = Pipeline(
        [
            BertNamedEntityRecognizer(),
            GraphRulesCharactersExtractor(),
            CoOccurrencesGraphExtractor(co_occurrences_dist),
        ]
    )

    no_corefs_out = no_corefs_pipeline(tokens=tokens, sentences=sentences)
    archive_pipeline_state(_run, no_corefs_out, "no_corefs_pipeline_state")

    # Gold pipeline using annotations
    # -------------------------------
    gold_pipeline = Pipeline([CoOccurrencesGraphExtractor(co_occurrences_dist)])

    gold_characters = get_thg_characters(tokens)
    gold_out = gold_pipeline(
        tokens=tokens, sentences=sentences, characters=gold_characters
    )

    # Comparison
    # ----------
    full_mapping, full_reverse_mapping = align_characters(
        gold_out.characters, full_out.characters
    )
    no_corefs_mapping, no_corefs_reverse_mapping = align_characters(
        gold_out.characters, no_corefs_out.characters
    )

    layout = shared_layout(
        gold_out.characters_graph,
        [full_out.characters_graph, no_corefs_out.characters_graph],
        [full_reverse_mapping, no_corefs_reverse_mapping],
    )
    archive_graph(_run, full_out, "full_thg", layout)
    archive_graph(_run, no_corefs_out, "no_corefs_thg", layout)
    archive_graph(_run, gold_out, "gold_thg", layout)

    for config, out, mapping in [
        ("full", full_out, full_mapping),
        ("no_corefs", no_corefs_out, no_corefs_mapping),
    ]:

        nodes_metrics = score_characters_extraction(
            [character.names for character in gold_out.characters],
            [character.names for character in out.characters],
        )
        print(f"{config} nodes metrics: {nodes_metrics}")
        for k, v in nodes_metrics.items():
            _run.log_scalar(f"{config}.nodes.{k}", v)

        edges_metrics = score_network_extraction_edges(
            gold_out.characters_graph, out.characters_graph, mapping
        )
        print(f"{config} edges metrics: {edges_metrics}")
        for k, v in edges_metrics.items():
            _run.log_scalar(f"{config}.edges.{k}", v)
