from typing import List, Set
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def score_characters_extraction(refs: List[Set[str]], preds: List[Set[str]]) -> dict:

    precision_scores = [[0.0] * len(refs)] * len(preds)
    for r_i, ref in enumerate(refs):
        for p_i, pred in enumerate(preds):
            precision_scores[p_i][r_i] = 1 - len(pred - ref) / len(pred)
    graph = csr_matrix(precision_scores)
    perm = maximum_bipartite_matching(graph)
    precision = sum(precision_scores[perm]) / len(preds)

    recall_scores = [[0.0] * len(refs)] * len(preds)
    for r_i, ref in enumerate(refs):
        for p_i, pred in enumerate(preds):
            recall_scores[p_i][r_i] = 1 if len(ref.intersection(pred)) > 0 else 0
    graph = csr_matrix(recall_scores)
    perm = maximum_bipartite_matching(graph)
    recall = sum(recall_scores[perm]) / len(refs)

    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }
