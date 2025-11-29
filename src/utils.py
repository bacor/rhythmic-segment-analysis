"""Miscellaneous helpers for paths, matplotlib styling, and figures."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DEFAULT_STYLE = CUR_DIR / "helvetica.mplstyle"

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_data_directory(subdirectory: Optional[str] = None) -> str:
    """Resolve the shared data directory from ``SHARED_DATA_DIR`` env var.

    Parameters
    ----------
    subdirectory:
        Optional subfolder to append to the base path.
    """
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = os.getenv("SHARED_DATA_DIR")
    if subdirectory:
        data_dir = os.path.join(data_dir, subdirectory)

    if not os.path.exists(data_dir):
        raise Exception(f"Data directory does not exist: {data_dir}")

    return data_dir


def show_readme(data_dir: Union[str, Path], filename: str = "README.md") -> None:
    """Render a README from the given data directory in a notebook.

    Parameters
    ----------
    data_dir:
        Root data directory containing the README.
    filename:
        Name of the README file to display.
    """
    from IPython.display import Markdown, display
    from pathlib import Path

    readme_path = Path(os.path.join(data_dir, filename))
    return display(Markdown(readme_path.read_text()))


def load_synthetic_intervals(
    path: Union[str, Path] = DATA_DIR / "synthetic-intervals.json",
) -> Any:
    """Load cached synthetic intervals from disk.

    Parameters
    ----------
    path:
        JSON file containing cached intervals.
    """
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def savefig(
    name: Union[str, Path], refresh: bool = False, dpi: int = 300, **kws: Any
) -> None:
    """Save a figure as PDF and PNG in ``figures/``.

    Parameters
    ----------
    name:
        File stem or path relative to ``figures/`` (extension is added).
    refresh:
        Force regeneration even if a file already exists.
    dpi:
        Resolution for the PNG output.
    **kws:
        Extra arguments forwarded to :func:`matplotlib.pyplot.savefig`.
    """
    base_path = (FIGURES_DIR / Path(name)).with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = base_path.with_suffix(".pdf")
    if refresh or not pdf_path.exists():
        plt.savefig(pdf_path, **kws)

    png_path = base_path.with_suffix(".png")
    if refresh or not png_path.exists():
        plt.savefig(png_path, dpi=dpi, **kws)


def subplot_title(
    index: int,
    title: str,
    ha: str = "left",
    x: float = 0,
    fontsize: int = 10,
    fontweight: str = "bold",
    ax: Optional[plt.Axes] = None,
    **title_kws: Any,
) -> plt.Text:
    """Prefix a subplot title with a lettered label (A., B., ...).

    Parameters
    ----------
    index:
        Subplot index used to pick the alphabetic prefix.
    title:
        Text to show after the prefix.
    ha, x, fontsize, fontweight, ax:
        Passed through to :meth:`matplotlib.axes.Axes.set_title`.
    **title_kws:
        Additional keyword arguments forwarded to ``set_title``.
    """
    if ax is None:
        ax = plt.gca()
    return ax.set_title(
        f"{ALPHABET[index]}. {title}",
        ha=ha,
        x=x,
        fontsize=fontsize,
        fontweight=fontweight,
        **title_kws,
    )


def subplots_grid(
    N: int,
    ratios: Tuple[float, float] = (3, 3),
    max_ncols: int = 3,
    **kws: Any,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a grid of subplots sized by ``ratios`` with limited columns.

    Parameters
    ----------
    N:
        Total number of panels to create.
    ratios:
        Width/height scaling applied to the figure size.
    max_ncols:
        Maximum number of columns before wrapping to a new row.
    **kws:
        Extra keyword arguments forwarded to :func:`matplotlib.pyplot.subplots`.
    """
    ncols = min(N, max_ncols)
    nrows = int(np.ceil(N / ncols))
    if "ncols" in kws:
        kws.pop("ncols")
    if "nrows" in kws:
        kws.pop("nrows")
    return plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(ncols * ratios[0], nrows * ratios[1]), **kws
    )


def set_mpl_style(style_path: Union[str, Path, None] = None) -> None:
    """Apply the repo's default Matplotlib style (Helvetica) or a custom one."""
    path = Path(style_path) if style_path is not None else DEFAULT_STYLE
    plt.style.use(path)


def set_figsize_cm(width_cm: float, height_cm: float) -> None:
    """Set the current figure size in centimeters."""
    plt.gcf().set_size_inches(width_cm / 2.54, height_cm / 2.54)


def get_line_props(line: Line2D) -> Dict[str, Union[str, float, int]]:
    """Extract a subset of line properties for reuse when redrawing."""
    return {
        "color": line.get_color(),
        "linestyle": line.get_linestyle(),
        "linewidth": line.get_linewidth(),
        "marker": line.get_marker(),
        "markerfacecolor": line.get_markerfacecolor(),
        "markeredgecolor": line.get_markeredgecolor(),
        "markeredgewidth": line.get_markeredgewidth(),
        "markersize": line.get_markersize(),
        "zorder": line.get_zorder(),
    }
