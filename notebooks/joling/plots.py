"""Charts for the joling notebooks. Plain matplotlib, white background, report-ready."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# (column in the summary table, legend label, colour, marker shape).
# Shape repeats what colour says, so the chart still reads in greyscale.
CONDITIONS = [
    ("val_auroc", "Validation (standard)", BLUE, "o"),
    ("val_layout_removed", "Layout removed", ORANGE, "s"),
    ("unseen_domain_mean", "Unseen domain", AQUA, "^"),
    ("unseen_layout_removed_mean", "Unseen domain + layout removed", YELLOW, "D"),
]


def _clean(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SECONDARY, length=0, labelsize=9)


def plot_stress_summary(table: pd.DataFrame, kaggle_col: str = "kaggle_public"):
    """One row per model; each marker is the AUROC under one test condition.

    ``table`` needs a ``model`` column plus the columns named in ``CONDITIONS``.
    If ``kaggle_col`` is present, the Kaggle public score is drawn as a black tick.
    """
    table = table.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 0.85 * len(table) + 1.6), facecolor=SURFACE)
    lanes = np.linspace(0.27, -0.27, len(CONDITIONS))   # small vertical offsets so equal values stay visible
    for i, row in table.iterrows():
        y = len(table) - 1 - i
        vals = [row[c] for c, *_ in CONDITIONS]
        if kaggle_col in table and pd.notna(row[kaggle_col]):
            vals.append(row[kaggle_col])
            ax.plot([row[kaggle_col]] * 2, [y - 0.36, y + 0.36], color=INK, linewidth=2,
                    label="Kaggle public (hidden test)" if i == 0 else None, zorder=3)
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=1.5, zorder=1)
        for lane, (col, label, colour, marker) in zip(lanes, CONDITIONS):
            ax.plot(row[col], y + lane, marker=marker, color=colour, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none",
                    label=label if i == 0 else None, zorder=4)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["model"][::-1], color=INK, fontsize=10)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.set_xlabel("AUROC on the shared 2,000-row validation set", color=SECONDARY, fontsize=9)
    _clean(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8.5,
              labelcolor=SECONDARY, handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout()
    return fig


def plot_domain_drop(per_domain: dict[str, pd.DataFrame], names: list[str]):
    """Seen vs. unseen AUROC for every domain, one panel per model (shared axes).

    ``per_domain``: {model name: per-domain table from ``domain_stress_test``}.
    ``names``: readable name of each domain, indexed by domain number.
    """
    n = len(per_domain)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 1.6, 0.42 * len(names) + 1.7), sharex=True,
                             sharey=True, facecolor=SURFACE, squeeze=False)
    order = next(iter(per_domain.values())).sort_values("drop")["domain"].tolist()  # biggest drop on top
    for ax, (model, df) in zip(axes[0], per_domain.items()):
        df = df.set_index("domain").loc[order]
        ys = np.arange(len(order))
        ax.hlines(ys, df["unseen"], df["seen"], color=GRID, linewidth=2, zorder=1)
        ax.plot(df["seen"], ys, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label="Domain seen in training", zorder=3)
        ax.plot(df["unseen"], ys, "^", color=ORANGE, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label="Domain held out of training", zorder=3)
        for y, (seen, unseen) in zip(ys, zip(df["seen"], df["unseen"])):
            ax.annotate(f"{unseen - seen:+.3f}", (min(seen, unseen), y), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=8, color=SECONDARY)
        ax.set_title(model, loc="left", fontsize=10, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([names[d] for d in order], color=INK, fontsize=9)
        ax.set_xlabel("Validation AUROC within the domain", color=SECONDARY, fontsize=9)
        _clean(ax)
    lo = min(df["unseen"].min() for df in per_domain.values())
    axes[0][0].set_xlim(lo - 0.035, 1.01)
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2, frameon=False, fontsize=8.5,
                      labelcolor=SECONDARY, handletextpad=0.3)
    fig.tight_layout()
    return fig
