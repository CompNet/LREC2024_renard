from typing import List, Tuple


def find_pattern(lst: list, pattern: list) -> List[Tuple[int, int]]:
    """Search all occurrences of pattern in lst.

    :return: a list of pattern coordinates.
    """
    coords = []
    for i in range(len(lst)):
        if lst[i] == pattern[0] and lst[i : i + len(pattern)] == pattern:
            coords.append((i, i + len(pattern)))
    return coords
