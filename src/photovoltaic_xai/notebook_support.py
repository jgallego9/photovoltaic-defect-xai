"""Find the checkout and display figures in the notebooks."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Image, display


def project_root(root=None):
    """Find the project from its root or a notebook working directory."""
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Open the notebook from the photovoltaic-xai project")


def decimal_label(value, position=None):
    return f"{value:g}".replace(".", ",")


def figure_style(font_size=10):
    return {
        "font.family": "DejaVu Sans",
        "font.size": font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def show_figure(fig, *, dpi=220, tight=True):
    """Display a figure with its intended dimensions, without writing to disk."""
    options = {"bbox_inches": "tight"} if tight else {}
    try:
        with BytesIO() as image:
            fig.savefig(image, format="png", dpi=dpi, facecolor="white", **options)
            display(Image(data=image.getvalue()))
    finally:
        plt.close(fig)
