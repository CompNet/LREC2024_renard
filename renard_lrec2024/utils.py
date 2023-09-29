from typing import List, Tuple
import os
from sacred.run import Run
from renard.pipeline.core import PipelineState


def find_pattern(lst: list, pattern: list) -> List[Tuple[int, int]]:
    """Search all occurrences of pattern in lst.

    :return: a list of pattern coordinates.
    """
    coords = []
    for i in range(len(lst)):
        if lst[i] == pattern[0] and lst[i : i + len(pattern)] == pattern:
            coords.append((i, i + len(pattern)))
    return coords


def archive_graph(_run: Run, state: PipelineState, graph_name: str):
    png_path = f"{graph_name}.png"
    state.plot_graph_to_file(png_path)
    _run.add_artifact(png_path)
    os.remove(png_path)

    gexf_path = f"{graph_name}.gexf"
    state.export_graph_to_gexf(gexf_path)
    _run.add_artifact(gexf_path)
    os.remove(gexf_path)
