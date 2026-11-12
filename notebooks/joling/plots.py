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


def plot_feature_by_class(features: pd.DataFrame, labels, columns: list[str], titles: list[str] | None = None):
    """Histogram of each chosen feature for human vs. AI texts (one small panel per feature)."""
    labels = np.asarray(labels)
    fig, axes = plt.subplots(1, len(columns), figsize=(4.3 * len(columns), 3.2), facecolor=SURFACE, squeeze=False)
    for ax, col, title in zip(axes[0], columns, titles or columns):
        lo, hi = np.percentile(features[col], [0.5, 99.5])
        bins = np.linspace(lo, hi, 45)
        for lab, name, colour in ((0, "Human", BLUE), (1, "AI", ORANGE)):
            ax.hist(features[col][labels == lab], bins=bins, histtype="stepfilled", alpha=0.35, color=colour,
                    edgecolor=colour, linewidth=1.6, label=name)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_yticks([])
        _clean(ax)
        ax.grid(False)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=SECONDARY, loc="upper left")
    fig.tight_layout()
    return fig


def plot_weight_curve(weights, scores, chosen: float):
    """Unseen-domain AUROC on the training rows for each blend weight; the chosen one is marked."""
    weights, scores = np.asarray(weights, float), np.asarray(scores, float)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    ax.plot(weights, scores, color=BLUE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    i = int(np.argmin(np.abs(weights - chosen)))
    ax.plot(weights[i], scores[i], "o", color=BLUE, markersize=11, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"chosen: {weights[i]:.1f}  ({scores[i]:.3f})", (weights[i], scores[i]), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.annotate(f"text model only\n{scores[0]:.3f}", (weights[0], scores[0]), xytext=(8, -4), textcoords="offset points",
                ha="left", va="top", fontsize=8.5, color=SECONDARY)
    ax.annotate(f"language-model\nfeatures only\n{scores[-1]:.3f}", (weights[-1], scores[-1]), xytext=(-8, -6),
                textcoords="offset points", ha="right", va="top", fontsize=8.5, color=SECONDARY)
    ax.set_xlabel("Weight of the language-model classifier in the blend", color=SECONDARY, fontsize=9)
    ax.set_ylabel("Unseen-domain AUROC\n(training rows)", color=SECONDARY, fontsize=9)
    ax.set_ylim(scores.min() - 0.012, scores.max() + 0.012)
    _clean(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_feature_by_group(features: pd.DataFrame, groups, columns: list[str], titles: list[str] | None = None,
                          order: list[str] | None = None):
    """Like ``plot_feature_by_class`` but for up to three named groups (fixed colour per position)."""
    groups = np.asarray(groups)
    order = order or sorted(set(groups))
    colours = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, len(columns), figsize=(4.3 * len(columns), 3.2), facecolor=SURFACE, squeeze=False)
    for ax, col, title in zip(axes[0], columns, titles or columns):
        lo, hi = np.percentile(features[col], [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        for name, colour in zip(order, colours):
            ax.hist(features[col][groups == name], bins=bins, density=True, histtype="stepfilled", alpha=0.3,
                    color=colour, edgecolor=colour, linewidth=1.6, label=name)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_yticks([])
        _clean(ax)
        ax.grid(False)
    axes[0][-1].legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY, loc="upper right")
    fig.tight_layout()
    return fig


def plot_learning_curve(sizes, train, val, title: str | None = None):
    """Training and validation AUROC against the number of training rows (log x-axis)."""
    sizes, train, val = np.asarray(sizes), np.asarray(train, float), np.asarray(val, float)
    fig, ax = plt.subplots(figsize=(6.6, 3.8), facecolor=SURFACE)
    for values, name, colour, marker in ((train, "Training", BLUE, "o"), (val, "Validation", ORANGE, "s")):
        ax.plot(sizes, values, color=colour, linewidth=2, marker=marker, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name)
        ax.annotate(f"{values[-1]:.4f}", (sizes[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=SECONDARY)
        ax.annotate(f"{values[0]:.4f}", (sizes[0], values[0]), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=SECONDARY)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] * 0.62, sizes[-1] * 1.5)
    ax.set_xlabel("Training rows (log scale)", color=SECONDARY, fontsize=9)
    ax.set_ylabel("AUROC", color=SECONDARY, fontsize=9)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    _clean(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY, loc="lower right")
    fig.tight_layout()
    return fig
