"""Helpers for plotting rhythmic pattern durations with Seaborn/Matplotlib."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import seaborn as sns

# Default placement for the horizontal inset colorbar.
DEFAULT_CBAR_INSET_AXES: Dict[str, Any] = {
    "width": "30%",
    "height": "4%",
    "loc": "upper right",
    "bbox_to_anchor": (0.05, 0.05, 0.9, 0.9),
}

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
)


DEFAULT_PAT_DUR_JOINT: Mapping[str, object] = dict(s=10, alpha=0.5)

DEFAULT_PAT_DUR_MARGINALS: Mapping[str, object] = dict(
    fill=True, common_norm=False, alpha=0.6, bw_adjust=0.5
)


def boost_legend_markers(ax=None, alpha=1.0, size=5):
    """Force legend markers to full opacity and a larger size."""
    ax = ax or plt.gca()
    legend = ax.get_legend()
    if legend is None:
        return ax

    for handle in legend.legend_handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(alpha)
        if size and hasattr(handle, "set_sizes"):
            handle.set_sizes([size])
        elif hasattr(handle, "set_markersize"):
            handle.set_markersize(size)

    return ax


def min_duration_boundary(
    duration: float, ax: Optional[plt.Axes] = None, **plot_kwargs
) -> plt.Axes:
    """Plot the minimum duration curve for a given pulse duration.

    The curve follows the reciprocal relationship `duration * min(p, 1 - p)`
    where `p` is the proportion of the pattern assigned to the first element.

    Parameters
    ----------
    duration:
        Reference duration used to construct the lower boundary curve.
    ax:
        Matplotlib axes to draw on; defaults to the current axes.
    **plot_kwargs:
        Additional keyword arguments forwarded to `Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the boundary was drawn on.
    """
    ax = ax or plt.gca()

    defaults = dict(
        color="0.8",
        linewidth=1,
        scalex=False,
        scaley=False,
        alpha=0.8,
        zorder=-10,
    )
    plot_args = {**defaults, **plot_kwargs}

    xs = np.linspace(0.01, 0.99, 200)
    ys = duration * np.where(xs <= 0.5, 1 / xs, 1 / (1 - xs))

    ax.plot(xs, ys, **plot_args)
    return ax


def pattern_duration_plot(
    rs,
    hue: Optional[str] = None,
    xlabel="Pattern",
    ylabel="Duration",
    marginals_kws=None,
    isochronous_axis=True,
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
        x=rs.patterns[:, 0], y=rs.durations, hue=hue_values, height=6, ratio=7
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


def annotate_ratio(
    i: int,
    j: int,
    pulse: float = 0.2,
    ax: Optional[plt.Axes] = None,
    marker_kws: Optional[Mapping[str, object]] = None,
    label_kws: Optional[Mapping[str, object]] = None,
) -> plt.Axes:
    """Mark a ratio `i:j` on a duration/pattern plot."""
    ax = ax or plt.gca()
    marker_args = dict(DEFAULT_RATIO_MARKER, **(marker_kws or {}))
    label_args = dict(DEFAULT_RATIO_LABEL, **(label_kws or {}))

    x = i / (i + j)
    y = (i + j) * pulse
    ax.plot(x, y, **marker_args)
    ax.annotate(f"{i}:{j}", xy=(x, y), **label_args)
    return ax


def annotate_ratios(
    pulse: float,
    max_factor: int = 10,
    length: int = 2,
    ratios: Optional[Iterable[Iterable[int]]] = None,
    **kwargs,
) -> None:
    """Annotate multiple integer ratios on the active axes.

    Parameters
    ----------
    pulse:
        Base pulse duration used to translate the ratio sum to a y-coordinate.
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
        annotate_ratio(*ratio, pulse=pulse, **kwargs)


def quantized_duration_ax(pulse: float, ax: plt.Axes) -> plt.Axes:
    """Set a y-axis locator/formatter so ticks fall on pulse multiples."""
    locator = mticker.MultipleLocator(base=pulse)

    def formatter(value, _pos):
        if value <= 0:
            return ""
        return str(int(round(value / pulse)))

    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))
    return ax


def quantized_dur_pat_plot(
    pulse: float,
    ax: plt.Axes,
    *,
    quantize_axis: bool = True,
    annotate: bool = True,
    pulse_boundary: bool = True,
    pulse_unit: str = "s",
    ylabel: Optional[str] = None,
) -> plt.Axes:
    """Apply common styling to a duration-vs-pattern axes."""
    if pulse_boundary:
        min_duration_boundary(pulse, ax=ax, linestyle="--")
    if annotate:
        annotate_ratios(pulse=pulse, ax=ax)
    if quantize_axis:
        quantized_duration_ax(pulse, ax)

    label = ylabel or f"Duration (pulse = {pulse:.2f}{pulse_unit})"
    ax.set_ylabel(label)
    return ax


def inset_cbar(
    norm: Normalize,
    cmap: str = "viridis",
    ax: Optional[Axes] = None,
    label: str = "Time",
    **inset_axes_kws: Any,
) -> Axes:
    """Draw a horizontal inset colorbar that matches a scatter hue.

    Parameters
    ----------
    norm:
        Matplotlib ``Normalize`` instance describing the color scale.
    cmap:
        Colormap name passed to ``ScalarMappable``.
    ax:
        Axis that hosts the scatter plot. Defaults to ``plt.gca()``.
    label:
        Label displayed on the colorbar axis.
    **inset_axes_kws:
        Extra keyword arguments forwarded to ``inset_axes`` to control the
        placement or size of the colorbar axes.

    Returns
    -------
    matplotlib.axes.Axes
        The newly created colorbar axes, useful for further customization.

    Examples
    --------
    Create a colorbar inset for a scatter plot using an explicit ``Normalize``:

    >>> from matplotlib import pyplot as plt
    >>> from matplotlib.colors import Normalize
    >>> fig, ax = plt.subplots()
    >>> _ = ax.scatter([0, 1], [0, 1], c=[0, 1], cmap="viridis")
    >>> cax = inset_cbar(Normalize(0, 1), ax=ax, label="Value")
    >>> isinstance(cax, plt.Axes)
    True
    >>> plt.close(fig)

    To derive ``norm`` from a ``RhythmicSegments`` instance use
    :func:`norm_from_segments`.

    """
    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    merged_inset_kwargs = DEFAULT_CBAR_INSET_AXES | inset_axes_kws
    cax = inset_axes(ax, bbox_transform=ax.transAxes, **merged_inset_kwargs)
    cbar = fig.colorbar(scalar_mappable, cax=cax, orientation="horizontal", label=label)
    cax.xaxis.set_ticks_position("bottom")
    return cax, cbar
