import os
import json
from pathlib import Path
from typing import Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "figures"
DEFAULT_STYLE = CUR_DIR / "helvetica.mplstyle"

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_data_directory(subdirectory=None):
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = os.getenv("SHARED_DATA_DIR")
    if subdirectory:
        data_dir = os.path.join(data_dir, subdirectory)

    if not os.path.exists(data_dir):
        raise Exception(f"Data directory does not exist: {data_dir}")

    return data_dir


def show_readme(data_dir, filename="README.md"):
    from IPython.display import Markdown, display
    from pathlib import Path

    readme_path = Path(os.path.join(data_dir, filename))
    return display(Markdown(readme_path.read_text()))


def load_synthetic_intervals(
    path: Union[str, Path] = DATA_DIR / "synthetic-intervals.json",
):
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def savefig(name: Union[str, Path], refresh: bool = False, dpi=300, **kws):
    base_path = (FIGURES_DIR / Path(name)).with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = base_path.with_suffix(".pdf")
    if refresh or not pdf_path.exists():
        plt.savefig(pdf_path, **kws)

    png_path = base_path.with_suffix(".png")
    if refresh or not png_path.exists():
        plt.savefig(png_path, dpi=dpi, **kws)


def subplot_title(
    index, title, ha="left", x=0, fontsize=10, fontweight="bold", ax=None, **title_kws
):
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


def subplots_grid(N, ratios=(3, 3), max_ncols=3, **kws):
    ncols = min(N, max_ncols)
    nrows = int(np.ceil(N / ncols))
    if "ncols" in kws:
        kws.pop("ncols")
    if "nrows" in kws:
        kws.pop("nrows")
    return plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(ncols * ratios[0], nrows * ratios[1]), **kws
    )


def set_mpl_style(style_path: Union[str, Path, None] = None):
    """Apply the repo's default Matplotlib style (Helvetica) or a custom one."""
    path = Path(style_path) if style_path is not None else DEFAULT_STYLE
    plt.style.use(path)


def set_figsize_cm(width_cm: float, height_cm: float):
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
