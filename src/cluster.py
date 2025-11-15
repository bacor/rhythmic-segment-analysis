from collections import Counter
from typing import Dict, Optional, Tuple
from sklearn.cluster import HDBSCAN
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes

from rhythmic_segments import RhythmicSegments

from .patdur import pattern_duration_plot, quantal_pattern_duration_plot

TransitionCounts = Counter[Tuple[str, str]]
ClusterStats = Dict[str, Dict[str, np.ndarray]]


def hdbscan_cluster(rs, min_cluster_size=20, seed=0, **kws):
    np.random.seed(seed)
    model = HDBSCAN(min_cluster_size=min_cluster_size, **kws)
    data = np.array([rs.patterns[:, 0], rs.durations]).T
    labels = model.fit_predict(data)
    rs.meta["label"] = [str(l) for l in labels]
    labeled = rs.query(f"label != '-1'")
    unlabeled = rs.query(f"label == '-1'")
    return rs, labeled, unlabeled


def count_value_transitions(
    rs: RhythmicSegments,
    column: str,
) -> TransitionCounts:
    """Count transitions between values in a metadata column"""
    counts: TransitionCounts = Counter()
    labels = rs.meta[column].to_numpy()
    for prev, nxt in zip(labels[:-1], labels[1:]):
        counts[(prev, nxt)] += 1
    return counts


def cluster_by_value(rs: RhythmicSegments, column: str) -> ClusterStats:
    """Aggregate segments by their value in a metadata column."""
    clusters: ClusterStats = {}
    for value in rs.meta[column].unique():
        subset = rs.filter(rs.meta[column] == value)
        points = np.column_stack((subset.patterns[:, :-1], subset.durations))
        clusters[value] = {
            "center": points.mean(axis=0),
            "std": points.std(axis=0),
            "points": points,
        }
    return clusters


def cluster_transition_network(
    rs: RhythmicSegments, column: str, min_transitions=0, ignore=["-1"]
) -> nx.DiGraph:
    """Build a directed graph with cluster centroids as nodes."""
    graph = nx.DiGraph()
    clusters = cluster_by_value(rs, column=column)
    for idx, (label, stats) in enumerate(clusters.items()):
        graph.add_node(label, pos=stats["center"], color=f"C{idx}")

    edge_counts = count_value_transitions(rs, column=column)
    max_count = max(edge_counts.values(), default=0)
    for (src, dst), count in edge_counts.items():
        if count < min_transitions:
            continue
        weight = 0.25 + 2.0 * (count / max_count) if max_count else 1.0
        graph.add_edge(src, dst, count=count, weight=weight)

    for label in ignore:
        graph.remove_node(label)
    return graph


def show_cluster_transition_network(
    rs: RhythmicSegments,
    column: str,
    min_transitions=0,
    ignore=["-1"],
    *,
    ax: Optional[Axes] = None,
    show_labels: bool = False,
) -> Axes:
    """Draw the cluster transition graph."""
    ax = ax or plt.gca()
    graph = cluster_transition_network(
        rs, column=column, min_transitions=min_transitions, ignore=ignore
    )
    pos = nx.get_node_attributes(graph, "pos")
    edge_weights = list(nx.get_edge_attributes(graph, "weight").values())

    if show_labels:
        bbox_style = dict(
            boxstyle="round, pad=0.3", facecolor="#fff", edgecolor="none", alpha=0.7
        )
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, bbox=bbox_style)

    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_weights,
        arrows=True,
        arrowsize=10,
        ax=ax,
        connectionstyle="arc3,rad=-0.10",
    )

    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    return ax


def clustered_pattern_duration_plot(rs, min_cluster_size=10, min_transitions=15):
    rs, labeled, unlabeled = hdbscan_cluster(rs, min_cluster_size=min_cluster_size)
    hue_order = labeled.meta["label"].value_counts().index
    g = pattern_duration_plot(labeled, hue="label", hue_order=hue_order, lw=0)

    # Mark unlabeled points in black
    g.ax_joint.plot(
        unlabeled.patterns[:, 0], unlabeled.durations, "kx", ms=3, alpha=0.5
    )

    # Transition network
    show_cluster_transition_network(rs, "label", min_transitions=min_transitions)

    # Legend bounds, etc
    g.ax_joint.legend_.remove()
    return g
