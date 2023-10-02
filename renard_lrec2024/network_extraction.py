from typing import Dict, Tuple, List, Set, Optional
import copy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from renard.pipeline.core import Mention
from renard.pipeline.characters_extraction import Character
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
        {"Flavius "},
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
        {"the boy from 4 "},
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

        for name in names:
            splitted = name.split(" ")
            mention_coords = find_pattern(tokens, splitted)
            for start, end in mention_coords:
                mentions.append(Mention(tokens[start:end], start, end))

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
) -> Dict[Character, Optional[Character]]:
    """Try to best align a set of predicted characters to a list of reference characters."""
    similarity = np.zeros((len(refs), len(preds)))
    for r_i, ref_character in enumerate(refs):
        for p_i, pred_character in enumerate(refs):
            intersection = ref_character.names.intersection(pred_character.names)
            union = ref_character.names.union(pred_character.names)
            similarity[r_i][p_i] = len(intersection) / len(union)

    graph = csr_matrix(similarity)
    perm = maximum_bipartite_matching(graph, perm_type="column")

    mapping = [[char, None] for char in refs]
    for r_i, mapping_i in enumerate(perm):
        if perm[r_i] != -1:
            mapping[r_i][1] = preds[mapping_i]

    return {c1: c2 for c1, c2 in mapping}
