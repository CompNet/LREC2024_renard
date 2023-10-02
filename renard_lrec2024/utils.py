from typing import List, Tuple
import os, pickle
from sacred.run import Run
import matplotlib.pyplot as plt
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


def archive_pipeline_state(_run: Run, state: PipelineState, name: str):
    pickle_path = f"{name}.pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump(state, f)
    _run.add_artifact(pickle_path)
    os.remove(pickle_path)


def archive_graph(_run: Run, state: PipelineState, graph_name: str):
    # PNG export
    png_path = f"{graph_name}.png"
    fig = plt.gcf()
    fig.set_size_inches(24, 24)
    fig.set_dpi(300)
    state.plot_graph_to_file(png_path, fig=fig)
    _run.add_artifact(png_path)
    os.remove(png_path)

    # Pickle export
    pickle_path = f"{graph_name}.pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump(state.characters_graph, f)
    _run.add_artifact(pickle_path)
    os.remove(pickle_path)

    # GEXF export
    gexf_path = f"{graph_name}.gexf"
    state.export_graph_to_gexf(gexf_path)
    _run.add_artifact(gexf_path)
    os.remove(gexf_path)
