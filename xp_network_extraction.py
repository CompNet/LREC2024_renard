from typing import Tuple, Union
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from rich import print
from renard.pipeline import Pipeline
from renard.pipeline.quote_detection import QuoteDetector
from renard.pipeline.ner import BertNamedEntityRecognizer
from renard.pipeline.corefs import BertCoreferenceResolver
from renard.pipeline.character_unification import GraphRulesCharacterUnifier
from renard.pipeline.speaker_attribution import BertSpeakerDetector
from renard.pipeline.graph_extraction import (
    CoOccurrencesGraphExtractor,
    ConversationalGraphExtractor,
)
from renard_lrec2024.character_unification import score_character_unification
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
    conversation_dist = (3, "sentences")


@ex.automain
def main(
    _run: Run,
    co_occurrences_dist: Union[int, Tuple[int, str]],
    conversation_dist: Union[int, Tuple[int, int]],
):
    tokens, sentences, _ = load_thg_bio(
        f"./data/network_extraction/TheHungerGames_annotated_no_extended_per_quotesfixed_chaptersfixed.conll"
    )

    # Full extraction pipeline
    # ------------------------
    full_pipeline = Pipeline(
        [
            BertNamedEntityRecognizer(),
            BertCoreferenceResolver(),
            GraphRulesCharacterUnifier(),
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
            GraphRulesCharacterUnifier(),
            CoOccurrencesGraphExtractor(co_occurrences_dist),
        ]
    )
    no_corefs_out = no_corefs_pipeline(tokens=tokens, sentences=sentences)
    archive_pipeline_state(_run, no_corefs_out, "no_corefs_pipeline_state")

    # conversational pipeline
    # -----------------------
    convers_pipeline = Pipeline(
        [
            QuoteDetector(),
            BertNamedEntityRecognizer(),
            GraphRulesCharacterUnifier(),
            BertSpeakerDetector(),
            ConversationalGraphExtractor(conversation_dist=conversation_dist),
        ]
    )
    convers_out = convers_pipeline(tokens=tokens, sentences=sentences)
    archive_pipeline_state(_run, convers_out, "convers_pipeline_state")

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
    convers_mapping, converse_reverse_mapping = align_characters(
        gold_out.characters, convers_out.characters
    )

    layout = shared_layout(
        gold_out.characters_graph,
        [
            full_out.characters_graph,
            no_corefs_out.characters_graph,
            convers_out.characters_graph,
        ],
        [full_reverse_mapping, no_corefs_reverse_mapping, converse_reverse_mapping],
    )
    archive_graph(_run, full_out, "full_thg", layout)
    archive_graph(_run, no_corefs_out, "no_corefs_thg", layout)
    archive_graph(_run, convers_out, "convers_out", layout)
    archive_graph(_run, gold_out, "gold_thg", layout)

    for config, out, mapping in [
        ("full", full_out, full_mapping),
        ("no_corefs", no_corefs_out, no_corefs_mapping),
        ("convers", convers_out, convers_mapping),
    ]:

        nodes_metrics = score_character_unification(
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
