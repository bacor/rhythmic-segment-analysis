from __future__ import annotations

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from rhythmic_segments import RhythmicSegments

_NON_NEGATIVE_TICK_FORMATTER = FuncFormatter(
    lambda value, _: f"{value:g}" if value >= 0 else ""
)
_SYMMETRIC_TICK_FORMATTER = FuncFormatter(
    lambda value, _: f"{abs(value):g}" if value <= 0 else f"{value:g}"
)


def _prepare_segments(
    rs: RhythmicSegments,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(rs.durations)
    shortest = rs.segments.min(axis=1)[order]
    longest = rs.segments.max(axis=1)[order]
    ys = np.arange(0, rs.count, 1)
    return shortest, longest, ys


def _resolve_limits(
    shortest: np.ndarray,
    longest: np.ndarray,
    left: float | None,
    right: float | None,
    symmetric: bool,
) -> Tuple[float, float]:
    auto_left = -float(shortest.max())
    auto_right = float(longest.max())
    if left is None:
        left = auto_left
    if right is None:
        right = auto_right
    if symmetric:
        span = max(abs(left), abs(right))
        left = -span
        right = span
    return left, right


def raster_plot(
    rs: RhythmicSegments,
    left: float | None = None,
    right: float | None = None,
    symmetric: bool = True,
    xlabel: str = "interval duration",
    s: float = 5,
    alpha: float = 0.25,
    marker: str = "o",
    **kwargs: Any,
) -> Axes:
    """Plot a combined raster on a single mirrored axis."""
    scatter_kws: Dict[str, Any] = {
        "s": s,
        "alpha": alpha,
        "marker": marker,
        "edgecolors": "none",
        **kwargs,
    }

    shortest, longest, ys = _prepare_segments(rs)
    left, right = _resolve_limits(shortest, longest, left, right, symmetric)

    ax = plt.gca()
    ax.set_xlim(left, right)
    ax.axvline(0, ymin=0, ymax=1, color="k", linewidth=1)

    ax.scatter(-shortest, ys, **scatter_kws)
    ax.scatter(longest, ys, **scatter_kws)
    ax.tick_params(axis="x")
    ax.xaxis.set_major_formatter(_SYMMETRIC_TICK_FORMATTER)

    ax.set_xlabel(xlabel, color="black")

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.yticks([])
    return ax


def twin_raster_plot(
    rs: RhythmicSegments,
    large_color: str = "C0",
    small_color: str = "C3",
    left: float | None = None,
    right: float | None = None,
    symmetric: bool = True,
    large_label: str = "larger interval",
    small_label: str = "smaller interval",
    s: float = 5,
    alpha: float = 0.25,
    marker: str = "o",
    **kwargs: Any,
) -> Tuple[plt.Axes, plt.Axes]:
    """Plot a raster of segment durations.

    Parameters
    ----------
    rs:
        A :class:`rhythmic_segments.RhythmicSegments` object that provides
        ``segments`` (interval pairs) and ``durations``.
    large_color, small_color:
        Matplotlib colors used for the larger and smaller intervals.
    left, right:
        Optional x-limits for the "larger interval" axis. If omitted, the limits
        are inferred from the data; ``symmetric=True`` forces them to be centered.
    symmetric:
        When ``True`` (default) both axes share the same magnitude but opposite sign.
    large_label, small_label:
        Axis labels for the large/small interval axes.
    s, alpha, marker:
        Marker style arguments passed through to :func:`matplotlib.pyplot.scatter`.
    **kwargs:
        Any additional keyword arguments forwarded to ``scatter``.

    Returns
    -------
    (Axes, Axes)
        Tuple containing the main axis (large intervals) and the secondary
        twin axis (small intervals) so the caller can further customise them.
    """
    scatter_kws: Dict[str, Any] = {
        "s": s,
        "alpha": alpha,
        "marker": marker,
        "edgecolors": "none",
        **kwargs,
    }

    shortest, longest, ys = _prepare_segments(rs)
    left, right = _resolve_limits(shortest, longest, left, right, symmetric)

    # Zero line
    plt.plot([0, 0], [ys[0], ys[-1]], "k--", lw=1)

    # Large intervals
    large_ax = plt.gca()
    large_ax.set_xlim(left, right)
    large_ax.scatter(longest, ys, color=large_color, **scatter_kws)
    large_ax.tick_params(axis="x", labelcolor=large_color)
    large_ax.xaxis.set_major_formatter(_NON_NEGATIVE_TICK_FORMATTER)
    large_ax.set_xlabel(large_label, color=large_color)
    large_ax.xaxis.set_label_coords(0.75, -0.08)

    # Small intervals
    small_ax = large_ax.twiny()
    small_ax.set_xlim(-left, -right)
    small_ax.tick_params(axis="x", labelcolor=small_color)
    small_ax.xaxis.set_major_formatter(_NON_NEGATIVE_TICK_FORMATTER)
    small_ax.set_xlabel(small_label, color=small_color)
    small_ax.xaxis.set_label_coords(0.25, 1.08)
    small_ax.scatter(shortest, ys, color=small_color, **scatter_kws)

    # Format the spines
    small_ax.spines["top"].set_color(small_color)
    small_ax.spines["top"].set_linewidth(1)
    small_ax.spines["bottom"].set_color(large_color)
    small_ax.spines["bottom"].set_linewidth(1)
    small_ax.spines["left"].set_linewidth(False)
    large_ax.spines["left"].set_linewidth(False)
    small_ax.spines["right"].set_visible(False)
    large_ax.spines["right"].set_visible(False)
    plt.yticks([])
    return large_ax, small_ax
