"""Helpers for plotting rhythmic pattern durations with Seaborn/Matplotlib."""

from __future__ import annotations

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Iterable, List, Optional, Mapping


DEFAULT_RATIO_MARKER: Mapping[str, object] = dict(
    marker="+",
    color="k",
    scalex=False,
    scaley=False,
    zorder=10,
)

DEFAULT_RATIO_LABEL: Mapping[str, object] = dict(
    xytext=(0, 4),
    textcoords="offset points",
    ha="center",
    va="bottom",
    color="k",
    fontsize=8,
    zorder=10,
    bbox=dict(
        boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8, pad=0.1
    ),
)


DEFAULT_PAT_DUR_JOINT: Mapping[str, object] = dict(s=10, alpha=0.5)

DEFAULT_PAT_DUR_MARGINALS: Mapping[str, object] = dict(
    fill=True, common_norm=False, alpha=0.6
)

DURATION_BOUND_DEFAULTS = {
    "color": "0.8",
    "linewidth": 1,
    "scalex": False,
    "scaley": False,
    "alpha": 0.8,
    "zorder": -10,
}


def annotate_ratio(
    i: int,
    j: int,
    quantum: float = 0.2,
    ax: Optional[plt.Axes] = None,
    marker_kws: Optional[Mapping[str, object]] = None,
    label_kws: Optional[Mapping[str, object]] = None,
) -> plt.Axes:
    """Mark a ratio `i:j` on a duration/pattern plot."""
    ax = ax or plt.gca()
    marker_args = dict(DEFAULT_RATIO_MARKER, **(marker_kws or {}))
    label_args = dict(DEFAULT_RATIO_LABEL, **(label_kws or {}))

    x = i / (i + j)
    y = (i + j) * quantum
    ax.plot(x, y, **marker_args)
    ax.annotate(f"{i}:{j}", xy=(x, y), **label_args)
    return ax


def annotate_ratios(
    quantum: float,
    max_factor: int = 10,
    length: int = 2,
    ratios: Optional[Iterable[Iterable[int]]] = None,
    **kwargs,
) -> None:
    """Annotate multiple integer ratios on the active axes.

    Parameters
    ----------
    quantum:
        Base quantum duration used to translate the ratio sum to a y-coordinate.
    max_factor:
        Largest integer to include in the automatically generated ratios.
    length:
        Length of the ratio tuple (defaults to binary ratios).
    ratios:
        Optional explicit list of ratios to annotate. When provided, the
        `max_factor`/`length` parameters are ignored.
    **kwargs:
        Additional keyword arguments forwarded to :func:`annotate_ratio`.
    """
    source = ratios or product(range(1, max_factor + 1), repeat=length)
    for ratio in source:
        annotate_ratio(*ratio, quantum=quantum, **kwargs)


def duration_boundary(
    duration: float,
    kind: str = "min",
    ax: Optional[Axes] = None,
    **plot_kwargs,
) -> List[Line2D]:
    """Plot the envelope that bounds feasible pattern durations.

    The curve follows the reciprocal relationship ``duration * min(p, 1 - p)``
    where ``p`` is the proportion of the pattern assigned to the first element.

    Parameters
    ----------
    duration:
        Reference duration used to construct the boundary curve.
    kind:
        Either ``"min"`` (default) to plot the minimal durations or ``"max"`` to
        plot the maximal durations implied by the reciprocal relationship.
    ax:
        Matplotlib axes to draw on; defaults to the current axes.
    **plot_kwargs:
        Additional keyword arguments forwarded to `Axes.plot`.

    Returns
    -------
    list[matplotlib.lines.Line2D]
        Matplotlib line objects created by ``Axes.plot``.
    """
    ax = ax or plt.gca()
    plot_args = {**DURATION_BOUND_DEFAULTS, **plot_kwargs}
    xs = np.linspace(0.01, 0.99, 200)
    if kind == "min":
        ys = np.where(xs <= 0.5, 1 / xs, 1 / (1 - xs))
    elif kind == "max":
        ys = np.where(xs > 0.5, 1 / xs, 1 / (1 - xs))
    else:
        raise ValueError("kind must be 'min' or 'max'")
    return ax.plot(xs, duration * ys, **plot_args)


def pattern_duration_plot(
    rs,
    hue: Optional[str] = None,
    hue_order=None,
    xlabel="pattern",
    ylabel="duration",
    marginals_kws=None,
    isochronous_axis=True,
    palette=None,
    **joint_kws,
) -> sns.JointGrid:
    """Create a joint scatter/KDE plot of rhythmic patterns vs durations.

    Parameters
    ----------
    rs:
        RhythmicSegments result (expects `.patterns`, `.durations`, and `.meta`).
    hue:
        Metadata key to use for coloring the scatter points. Leave `None` for a
        single-color plot.

    Returns
    -------
    seaborn.axisgrid.JointGrid
        Seaborn JointGrid object for further customization.
    """
    hue_values = rs.meta[hue] if hue else None
    grid = sns.JointGrid(
        x=rs.patterns[:, 0],
        y=rs.durations,
        hue=hue_values,
        hue_order=hue_order,
        height=6,
        ratio=7,
        palette=palette,
    )

    j_kws = dict(DEFAULT_PAT_DUR_JOINT, **(joint_kws or {}))
    grid.plot_joint(sns.scatterplot, **j_kws)

    marg_kws = dict(DEFAULT_PAT_DUR_MARGINALS, **(marginals_kws or {}))
    grid.plot_marginals(sns.kdeplot, **marg_kws)
    grid.set_axis_labels(xlabel=xlabel, ylabel=ylabel)

    # Mark isochrony
    if isochronous_axis:
        plt.sca(grid.ax_joint)
        plt.axvline(0.5, c="0.8", lw=1, zorder=-1, ls=":")
    return grid


def quantal_duration_ax(quantum: float, ax: plt.Axes) -> plt.Axes:
    """Set a y-axis locator/formatter so ticks fall on quantum multiples."""
    locator = mticker.MultipleLocator(base=quantum)

    def formatter(value, _pos):
        if value <= 0:
            return ""
        return str(int(round(value / quantum)))

    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))
    return ax


def quantal_pattern_duration_plot(
    quantum: float,
    ax: plt.Axes,
    *,
    quantal_axis: bool = True,
    annotate: bool = True,
    quantum_boundary: bool = True,
    quantum_unit: str = "s",
    ylabel: Optional[str] = None,
    annotate_ratios_kws={},
) -> plt.Axes:
    """Apply common styling to a duration-vs-pattern axes."""
    if quantum_boundary:
        duration_boundary(quantum, kind="min", ax=ax, linestyle="--")
    if annotate:
        annotate_ratios(quantum=quantum, ax=ax, **annotate_ratios_kws)
    if quantal_axis:
        quantal_duration_ax(quantum, ax)

    label = ylabel or f"duration (quantum = {quantum:.2f}{quantum_unit})"
    ax.set_ylabel(label)
    return ax
