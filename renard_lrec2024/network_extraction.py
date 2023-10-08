from typing import Dict, Tuple, List, Set, Optional, Any
import copy
import networkx as nx
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from renard.pipeline.core import Mention
from renard.pipeline.characters_extraction import Character
from renard.plot_utils import layout_nx_graph_reasonably
from renard_lrec2024.utils import find_pattern


def case_variations(name_set: Set[str]) -> Set[str]:
    new_name_set = copy.copy(name_set)
    for name in name_set:
        new_name_set.add(name.capitalize())
    return new_name_set


THG_CHARACTERS_NAMES = [
    case_variations(nameset)
    for nameset in [
        {
            "Katniss",
            "Catnip",
            "Sweetheart",
            "girl on fire",
            "Dr. Everdeen",
            "Fire Girl",
            "I",
        },
        {"my sister", "Prim", "my little sister", "little duck", "Primrose"},
        {"our mother", "My mother"},
        {"Greasy Sae", "Greasy Sac"},  # typo
        {"Mayor Undersee", "the mayor", "Madge's father"},
        {"Madge", "the mayor's daughter"},
        {"Haymitch", "our mentor"},
        {
            "Peeta",
            "the baker's son",
            "the boy with the bread",
            "Lover Boy",
        },
        {"the baker's wife", "the witch"},
        {"the baker", "Peeta Mellark's father", "Peeta's father"},
        {"Rooba", "the butcher"},
        {"President Snow", "The president"},
        {
            "the redheaded Avox girl",
            "the redheaded girl",
            "the girl with the red hair",
        },
        {
            "Cato",
            "The boy from District 2",
            "the brutish boy from District 2",
        },
        {"Rue", "The little girl from District 11"},
        {
            "Glimmer",
            "the girl tribute from District 1",
            "District one girl",
            "the girl from District 1",
        },
        {
            "Foxface",
            "the fox-faced girl from District 5",
            "the fox-faced girl",
            "the girl from five",
        },
        {"the boy from Ten", "the crippled boy from 10"},
        {
            "Clove",
            "the girl from District 2",
            "the girl from district two",
        },
        {
            "the boy from district 1",
            "The boy from District 1",
            "The boy from 1",
        },
        {
            "the boy from District 3",
            "The boy from Three",
            "The boy from District Three",
        },
        {"Career Tributes", "Careers", "Career Pack"},
        {"My father"},
        {"Gale"},
        {"Effie Trinket"},
        {"Peacekeepers"},
        {"Head Peacekeeper"},
        {"Flavius"},
        {"Venia"},
        {"Octavia"},
        {"Cinna"},
        {"Portia"},
        {"Delly Cartwright"},
        {"Gamemakers"},
        {"Atala"},
        {"Caesar Flickerman", "Caesar"},
        {"Gamemaker"},
        {"the girl from District 10"},
        {"Titus"},
        {"the boy from District 9"},
        {"the girl from District 3"},
        {"the boy from 4"},
        {"the boy from District 5"},
        {"Thresh"},
        {"the girl from District 4"},
        {"Martin"},
        {"the girl from District 8"},
        {"Goat Man"},
        {"Buttercup"},
        {"Lady"},
        {"Johanna Mason"},
    ]
]


def get_thg_characters(tokens: List[str]) -> Set[Character]:

    characters = set()

    for names in THG_CHARACTERS_NAMES:

        mentions = []
        visited_mention_coords = []

        # we start by the longest names. We pick the largest patterns,
        # and then prevents the smallest one from overlapping.
        for name in reversed(sorted(names, key=len)):

            splitted = name.split(" ")
            mention_coords = find_pattern(tokens, splitted)

            for start, end in mention_coords:

                # check if the mention is overlapping with an already
                # visited one
                if any(
                    (start >= o_start and start <= o_end)
                    or (end >= o_start and end <= o_end)
                    for o_start, o_end in visited_mention_coords
                ):
                    continue

                mentions.append(Mention(tokens[start:end], start, end))
                visited_mention_coords.append((start, end))

        characters.add(Character(names, mentions))

    return characters


def split_thg_into_sentence(tokens: List[str]) -> List[List[str]]:
    sentences = []
    current_sentence = []
    in_quote = False

    for token in tokens:
        current_sentence.append(token)
        if not in_quote and token in [".", "?", "!"]:
            sentences.append(current_sentence)
            current_sentence = []
        if token == '"':
            in_quote = not in_quote

    if not len(current_sentence) == 0:
        sentences.append(current_sentence)

    return sentences


def load_thg_bio(path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    :param cut_into_chapters: if ``True``, return ``chapter_tokens``
        instead of ``tokens``

    :return: ``(tokens, bio_tags)`` or ``(chapter_tokens, bio_tags)``
    """
    tokens = []
    bio_tags = []

    with open(path) as f:
        for line in f:
            if line.isspace():
                continue
            token, tag = line.rstrip("\n").split("\t")
            tokens.append(token)
            bio_tags.append(tag)

    return (tokens, split_thg_into_sentence(tokens), bio_tags)


def align_characters(
    refs: List[Character], preds: List[Character]
) -> Tuple[Dict[Character, Optional[Character]], Dict[Character, Optional[Character]]]:
    """Try to best align a set of predicted characters to a list of
    reference characters.

    :return: a tuple:

            - a dict with keys from ``refs`` and values from
              ``preds``.  Values can be ``None``.

            - a dict with keys from ``preds`` and values from
              ``refs``.  Values can be ``None``.
    """
    similarity = np.zeros((len(refs), len(preds)))
    for r_i, ref_character in enumerate(refs):
        for p_i, pred_character in enumerate(preds):
            intersection = ref_character.names.intersection(pred_character.names)
            union = ref_character.names.union(pred_character.names)
            similarity[r_i][p_i] = len(intersection) / len(union)

    graph = csr_matrix(similarity)
    perm = maximum_bipartite_matching(graph, perm_type="column")

    mapping = [[char, None] for char in refs]
    for r_i, mapping_i in enumerate(perm):
        if perm[r_i] != -1:
            mapping[r_i][1] = preds[mapping_i]

    mapping = {c1: c2 for c1, c2 in mapping}

    reverse_mapping = {v: k for k, v in mapping.items()}
    for character in preds:
        if not character in reverse_mapping:
            reverse_mapping[character] = None

    return (mapping, reverse_mapping)


def score_network_extraction_edges(
    gold_graph: nx.Graph,
    pred_graph: nx.Graph,
    mapping: Dict[Character, Optional[Character]],
) -> dict:
    recall_list = []
    for r1, r2 in gold_graph.edges:
        c1 = mapping[r1]
        c2 = mapping[r2]
        if (c1, c2) in pred_graph.edges:
            recall_list.append(1)
        else:
            recall_list.append(0)
    recall = sum(recall_list) / len(recall_list)

    # edge precision
    precision_list = []
    reverse_mapping = {v: k for k, v in mapping.items() if not v is None}
    for c1, c2 in pred_graph.edges:
        r1 = reverse_mapping.get(c1)
        r2 = reverse_mapping.get(c2)
        if (r1, r2) in gold_graph.edges:
            precision_list.append(1)
        else:
            precision_list.append(0)
    precision = sum(precision_list) / len(precision_list)

    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def shared_layout(
    G: nx.Graph,
    H_list: List[nx.Graph],
    H_to_G_list: List[Dict[Character, Optional[Character]]],
) -> Dict[Character, np.ndarray]:
    """Compute a layout shared between a reference graph G and a list
    of related graphs H.

    :param G: a reference graph with Character nodes
    :param H_list: a list of related graphs with Character nodes
    :param H_to_G_list: A list of mapping from V_H to V_G ∪ {∅}

    :return: a dict of form ``{character: (x, y)}``
    """
    # Extract the union of both graph nodes, considering the mapping
    # between G and H.
    nodes = list(G.nodes)
    for H, H_to_G in zip(H_list, H_to_G_list):
        for H_node, G_node in H_to_G.items():
            if G_node is None:
                nodes.append(H_node)

    # We do something similar to above here, except we define the
    # following mapping g_E from E_H to E_G:
    # { (g_V(n1), g_V(n2)) if g_V(n1) ≠ ∅ and g_V(n2) ≠ ∅,
    # { ∅                  otherwise
    edges = list(G.edges)
    for H, H_to_G in zip(H_list, H_to_G_list):
        for n1, n2 in H.edges:
            if H_to_G[n1] is None or H_to_G[n2] is None:
                n1 = H_to_G.get(n1) or n1
                n2 = H_to_G.get(n2) or n2
                edges.append((n1, n2))

    # Construct the union graph J between G and H
    J = nx.Graph()
    for node in nodes:
        J.add_node(node)
    for edge in edges:
        J.add_edge(*edge)

    # union graph layout
    # layout = layout_nx_graph_reasonably(J)
    layout = nx.kamada_kawai_layout(J)

    # the layout will be used for both G and all graphs in
    # H_list. However, some nodes from these H are not in the layout
    # dictionary: only their equivalent in G are here. We add these
    # nodes now, by specifying that their positions is the same as the
    # position of their equivalent nodes in G
    for H, H_to_G in zip(H_list, H_to_G_list):
        for node in H.nodes:
            if not node in layout:
                layout[node] = layout[H_to_G[node]]

    return layout
