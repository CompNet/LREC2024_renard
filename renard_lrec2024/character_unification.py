from typing import List, Set
from more_itertools import flatten
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from renard.pipeline import PipelineStep
from renard.pipeline.ner import NEREntity
from renard_lrec2024.utils import find_pattern


class PDNCPerfectNamedEntityRecognizer(PipelineStep):
    """
    Simulate a NER model that only extract mentions from the
    characters included in ``refs``.
    """

    def __call__(self, tokens: List[str], refs: List[Set[str]], **kwargs) -> dict:

        entities = []

        for name in flatten(refs):

            coords_list = find_pattern(tokens, name.split(" "))

            for coords in coords_list:
                entities.append(
                    NEREntity(
                        tokens[coords[0] : coords[1]], coords[0], coords[1], "PER"
                    )
                )

        return {"entities": entities}

    def needs(self):
        return {"tokens", "refs"}

    def production(self):
        return {"entities"}


def score_character_unification(refs: List[Set[str]], preds: List[Set[str]]) -> dict:

    preds_nb = len(preds)
    refs_nb = len(refs)

    precision_scores = np.zeros((preds_nb, refs_nb))
    for r_i, ref in enumerate(refs):
        for p_i, pred in enumerate(preds):
            precision_scores[p_i][r_i] = 1 - len(pred - ref) / len(pred)
    graph = csr_matrix(precision_scores)
    perm = maximum_bipartite_matching(graph, perm_type="column")
    # sum only the pair of name sets that maps together
    mapped_precision_scores = np.take_along_axis(precision_scores, perm[:, None], 1)
    mapped_precision_scores = mapped_precision_scores.flatten()
    # ignore unmapped sets
    mapped_precision_scores *= (perm != -1).astype(int)
    precision = sum(mapped_precision_scores) / preds_nb

    recall_scores = np.zeros((preds_nb, refs_nb))
    for r_i, ref in enumerate(refs):
        for p_i, pred in enumerate(preds):
            recall_scores[p_i][r_i] = 1 if len(ref.intersection(pred)) > 0 else 0
    graph = csr_matrix(recall_scores)
    perm = maximum_bipartite_matching(graph, perm_type="column")
    # sum only the pair of name sets that maps together
    mapped_recall_scores = np.take_along_axis(recall_scores, perm[:, None], 1)
    mapped_recall_scores = mapped_recall_scores.flatten()
    # ignore unmapped sets
    mapped_recall_scores *= (perm != -1).astype(int)
    recall = sum(mapped_recall_scores) / refs_nb

    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }
