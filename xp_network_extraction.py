from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from rich import print
from renard.pipeline import Pipeline
from renard.pipeline.ner import BertNamedEntityRecognizer, ner_entities
from renard.pipeline.corefs import BertCoreferenceResolver
from renard.pipeline.characters_extraction import GraphRulesCharactersExtractor
from renard.pipeline.graph_extraction import CoOccurencesGraphExtractor
from renard_lrec2024.characters_extraction import score_characters_extraction
from renard_lrec2024.network_extraction import load_thg_bio, get_thg_characters
from renard_lrec2024.utils import archive_graph


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore


@ex.config
def config():
    pass


@ex.automain
def main(_run: Run):

    tokens, sentences, gold_bio_tags = load_thg_bio(
        f"./data/network_extraction/TheHungerGames_annotated_no_extended_per_quotesfixed_chaptersfixed.conll"
    )
    gold_entities = ner_entities(tokens, gold_bio_tags)
    gold_characters = get_thg_characters(tokens)

    # Full extraction pipeline (not using annotations)
    # ------------------------------------------------
    pipeline = Pipeline(
        [
            BertNamedEntityRecognizer(),
            BertCoreferenceResolver(),
            GraphRulesCharactersExtractor(),
            CoOccurencesGraphExtractor(),
        ]
    )

    out = pipeline(tokens=tokens, sentences=sentences)

    # Gold pipeline using annotations
    # -------------------------------
    gold_pipeline = Pipeline(
        [GraphRulesCharactersExtractor(), CoOccurencesGraphExtractor()]
    )

    gold_out = gold_pipeline(tokens=tokens, sentences=sentences, entities=gold_entities)

    # Comparison
    # ----------
    archive_graph(_run, out, "thg")
    archive_graph(_run, out, "thg_gold")

    nodes_metrics = score_characters_extraction(
        [character.names for character in gold_out.characters],
        [character.names for character in out.characters],
    )
    print(f"nodes metrics: {nodes_metrics}")
