"""Figures shared by the notebooks.

Style rules: one fixed colour per model, in the same order on every plot, so a model keeps
its colour across figures. Marks are thin, grids are light, and there is a single y-axis
per chart. For more than one series a legend is always shown.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geo import to_local_xy_m

#: Fixed categorical order (validated colour-blind-safe for adjacent series).
MODEL_COLORS = {
    "k-NN": "#2a78d6", "CNN": "#eb6834", "SVR": "#1baf7a", "ANN": "#eda100",
    "XGBoost": "#e87ba4", "LightGBM": "#008300", "Hybrid": "#4a3aa7",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style():
    """Apply a clean, print-friendly matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "font.size": 10, "lines.linewidth": 2,
    })


def cdf(errors: dict[str, np.ndarray], ax=None, xmax: float = 2000, thresholds=None):
    """Empirical CDF of localization error, one line per model (paper Fig. 3)."""
    ax = ax or plt.subplots(figsize=(7, 4.2))[1]
    for name, e in errors.items():
        e = np.sort(e)
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), label=name,
                color=MODEL_COLORS.get(name))
    for t in thresholds or ():
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set(xlim=(0, xmax), ylim=(0, 1), xlabel="Localization error (m)",
           ylabel="Fraction of test samples", title="Error CDF")
    ax.legend(loc="lower right")
    return ax


def compare_bars(ours: pd.Series, paper: pd.Series, ax=None, title="Mean error: ours vs. paper"):
    """Grouped bars: reproduced mean error next to the paper's, per model."""
    ax = ax or plt.subplots(figsize=(7.5, 4))[1]
    names = [m for m in MODEL_COLORS if m in ours.index]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, [paper.get(m, np.nan) for m in names], w, color="#c3c2b7",
           label="Paper", edgecolor="white", linewidth=1)
    ax.bar(x + w / 2 + 0.01, [ours[m] for m in names], w,
           color=[MODEL_COLORS[m] for m in names], label="This reproduction",
           edgecolor="white", linewidth=1)
    for i, m in enumerate(names):
        ax.text(x[i] + w / 2 + 0.01, ours[m] + 4, f"{ours[m]:.0f}", ha="center",
                va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, names)
    ax.set(ylabel="Mean error (m)", title=title)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c3c2b7", label="Paper"),
                       Patch(color=MUTED, label="This reproduction (model colour)")],
              loc="upper right")
    return ax


def error_heatmap(y_true_deg: np.ndarray, errors_m: np.ndarray, cell_m: float = 150.0,
                  ax=None, vmax: float = 1000, min_count: int = 3):
    """Mean error per ``cell_m`` x ``cell_m`` grid cell (paper Fig. 4).

    Cells with fewer than ``min_count`` test samples are left blank.
    """
    ax = ax or plt.subplots(figsize=(7, 5.5))[1]
    x, y = to_local_xy_m(y_true_deg[:, 0], y_true_deg[:, 1])
    xb = np.floor((x - x.min()) / cell_m).astype(int)
    yb = np.floor((y - y.min()) / cell_m).astype(int)
    grid_sum = np.zeros((yb.max() + 1, xb.max() + 1))
    grid_n = np.zeros_like(grid_sum)
    np.add.at(grid_sum, (yb, xb), errors_m)
    np.add.at(grid_n, (yb, xb), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(grid_n >= min_count, grid_sum / grid_n, np.nan)
    extent = (0, mean.shape[1] * cell_m / 1000, 0, mean.shape[0] * cell_m / 1000)
    im = ax.imshow(mean, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=vmax)
    ax.set(xlabel="East (km)", ylabel="North (km)",
           title=f"Mean error per {cell_m:.0f} m cell")
    ax.grid(False)
    plt.colorbar(im, ax=ax, label="Mean error (m)", shrink=0.85)
    valid = mean[~np.isnan(mean)]
    return ax, {f"cells_below_{t}m": float((valid < t).mean()) for t in (250, 300, 350)}
