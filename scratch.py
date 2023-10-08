# -*- eval: (code-cells-mode); -*-

# %%
import pickle

with open("./runs_network_extraction/3/full_thg.pickle", "rb") as f:
    G_full = pickle.load(f)

with open("./runs_network_extraction/3/no_corefs_thg.pickle", "rb") as f:
    G_no_corefs = pickle.load(f)

with open("./runs_network_extraction/3/gold_thg.pickle", "rb") as f:
    G_gold = pickle.load(f)

# %%
with open("./runs_network_extraction/2/full_pipeline_state.pickle", "rb") as f:
    full_state = pickle.load(f)

with open("./runs_network_extraction/2/no_corefs_pipeline_state.pickle", "rb") as f:
    no_corefs_state = pickle.load(f)


# %%
import matplotlib.pyplot as plt
from renard.pipeline import Pipeline
from renard.pipeline.ner import BertNamedEntityRecognizer
from renard.pipeline.corefs import BertCoreferenceResolver
from renard.pipeline.characters_extraction import GraphRulesCharactersExtractor
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor

with open("./runs_network_extraction/1/pipeline_state.pickle", "rb") as f:
    state = pickle.load(f)


pipeline = Pipeline(
    [
        BertNamedEntityRecognizer(),
        BertCoreferenceResolver(),
        GraphRulesCharactersExtractor(),
        CoOccurrencesGraphExtractor((10, "tokens")),
    ]
)
pipeline._pipeline_init_steps()
# test: cut coref resolution
state.corefs = None
out = pipeline.rerun_from(state, GraphRulesCharactersExtractor)


# also have a correct fig
fig = plt.gcf()
fig.set_dpi(300)
fig.set_size_inches(24, 24)
out.plot_graph(fig=fig)
plt.show()

# %%
from renard_lrec2024.characters_extraction import score_characters_extraction

nodes_metrics = score_characters_extraction(
    [character.names for character in G_gold.nodes],
    [character.names for character in out.characters],
)
print(nodes_metrics)

# %%
from typing import *
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from renard.pipeline.characters_extraction import Character
from renard_lrec2024.network_extraction import align_characters


def character_name_count(character: Character) -> Dict[str, int]:
    c = Counter([" ".join(mention.tokens) for mention in character.mentions])
    c = {c: count for c, count in c.items() if c in character.names}
    return c


def weighted_align_characters(
    refs: List[Character], preds: List[Character]
) -> Dict[Character, Optional[Character]]:
    """Try to best align a set of predicted characters to a list of reference characters.

    :return: a dict with keys from ``refs`` and values from ``preds``. Values can be ``None``.
    """
    similarity = np.zeros((len(refs), len(preds)))

    for r_i, ref_character in enumerate(refs):

        for p_i, pred_character in enumerate(preds):

            ref_count = character_name_count(ref_character)
            pred_count = character_name_count(pred_character)

            intersection = ref_character.names.intersection(pred_character.names)
            intersection_sum = sum(
                min(ref_count.get(name, 0), pred_count.get(name, 0))
                for name in intersection
            )

            union = ref_character.names.union(pred_character.names)
            union_sum = sum(
                max(ref_count.get(name, 0), pred_count.get(name, 0)) for name in union
            )

            similarity[r_i][p_i] = intersection_sum / union_sum

    graph = csr_matrix(similarity)
    perm = maximum_bipartite_matching(graph, perm_type="column")

    mapping = [[char, None] for char in refs]
    for r_i, mapping_i in enumerate(perm):
        if perm[r_i] != -1:
            mapping[r_i][1] = preds[mapping_i]

    return {c1: c2 for c1, c2 in mapping}


mapping = weighted_align_characters(list(G_gold.nodes), list(G.nodes))


# %%
from renard_lrec2024.network_extraction import align_characters

# edge recall
characters_mapping = align_characters(list(G_gold.nodes), list(G.nodes))
recall_list = []
for r1, r2 in G_gold.edges:
    c1 = characters_mapping[r1]
    c2 = characters_mapping[r2]
    if (c1, c2) in G.edges:
        recall_list.append(1)
    else:
        recall_list.append(0)
recall = sum(recall_list) / len(recall_list)
print(f"edges recall: {recall}")

# edge precision
precision_list = []
r_characters_mapping = {v: k for k, v in characters_mapping.items() if not v is None}
for c1, c2 in G.edges:
    r1 = r_characters_mapping.get(c1)
    r2 = r_characters_mapping.get(c2)
    if (r1, r2) in G_gold.edges:
        precision_list.append(1)
    else:
        precision_list.append(0)
precision = sum(precision_list) / len(precision_list)
print(f"edges precision: {precision}")

f1 = 2 * precision * recall / (precision + recall)
print(f"edges F1: {f1}")
