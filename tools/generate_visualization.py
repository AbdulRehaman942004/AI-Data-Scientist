import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langchain_core.tools import tool

OUTPUT_DIR = "outputs"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _apply_style(ax, bg="#0d1117", border="#30363d", muted="#8b949e"):
    ax.set_facecolor(bg)
    ax.tick_params(colors=muted, labelsize=9)
    ax.xaxis.label.set_color(muted)
    ax.yaxis.label.set_color(muted)
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor(border)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=100, facecolor="#1c2128")
    plt.close(fig)


@tool
def generate_visualization(
    file_path: str,
    chart_type: str,
    column: str = "",
    column2: str = "",
) -> dict:
    """Generate and save a visualization for the dataset.

    Supported chart_type values:
    - 'histogram' : Distribution of a numeric column with KDE. Requires 'column'.
    - 'boxplot'   : Box plot to show spread and outliers. Requires 'column'.
    - 'violin'    : Violin plot showing full distribution shape. Requires 'column'.
                    Optional 'column2' (a low-cardinality categorical) to group by.
    - 'bar'       : Value counts of a categorical column (top 15). Requires 'column'.
    - 'heatmap'   : Correlation matrix of all numeric columns. No column needed.
    - 'scatter'   : Scatter plot between two numeric columns. Requires 'column' and 'column2'.
    - 'pairplot'  : Pairwise scatter matrix for numeric columns (max 5). No column needed.

    Returns the file path of the saved plot.
    """
    try:
        df = pd.read_csv(file_path)
        _ensure_output_dir()

        BG = "#1c2128"
        PANEL = "#0d1117"
        BLUE = "#1f6feb"
        PURPLE = "#8b5cf6"
        GREEN = "#3fb950"
        MUTED = "#8b949e"
        TEXT = "#e6edf3"

        sns.set_theme(style="darkgrid")
        plt.rcParams.update({
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "grid.color": "#21262d",
            "grid.linewidth": 0.6,
        })

        # ── histogram ─────────────────────────────────────────────────────────
        if chart_type == "histogram":
            if not column:
                return {"error": "histogram requires a 'column' argument"}
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor(BG)
            _apply_style(ax)
            sns.histplot(df[column].dropna(), kde=True, ax=ax, color=BLUE, alpha=0.78, edgecolor="none")
            ax.set_title(f"Distribution of {column}", fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel(column)
            out_path = os.path.join(OUTPUT_DIR, f"histogram_{column}.png")
            _save(fig, out_path)

        # ── boxplot ───────────────────────────────────────────────────────────
        elif chart_type == "boxplot":
            if not column:
                return {"error": "boxplot requires a 'column' argument"}
            fig, ax = plt.subplots(figsize=(6, 7))
            fig.patch.set_facecolor(BG)
            _apply_style(ax)
            sns.boxplot(y=df[column].dropna(), ax=ax, color=PURPLE, linewidth=1.2,
                        flierprops={"marker": "o", "markerfacecolor": "#f85149", "markersize": 4})
            ax.set_title(f"Box Plot of {column}", fontsize=13, fontweight="bold", pad=10)
            out_path = os.path.join(OUTPUT_DIR, f"boxplot_{column}.png")
            _save(fig, out_path)

        # ── violin ────────────────────────────────────────────────────────────
        elif chart_type == "violin":
            if not column:
                return {"error": "violin requires a 'column' argument"}
            if column2 and column2 in df.columns:
                # Grouped violin by categorical column2
                n_cats = df[column2].nunique()
                fig_w = max(9, n_cats * 1.6)
                fig, ax = plt.subplots(figsize=(fig_w, 6))
                fig.patch.set_facecolor(BG)
                _apply_style(ax)
                palette = sns.color_palette("husl", n_cats)
                sns.violinplot(
                    x=df[column2].astype(str), y=df[column], ax=ax,
                    palette=palette, inner="box", linewidth=1.0,
                )
                ax.set_title(f"Distribution of {column}  by  {column2}",
                             fontsize=13, fontweight="bold", pad=10)
                plt.xticks(rotation=30, ha="right")
                out_path = os.path.join(OUTPUT_DIR, f"violin_{column}_by_{column2}.png")
            else:
                fig, ax = plt.subplots(figsize=(6, 7))
                fig.patch.set_facecolor(BG)
                _apply_style(ax)
                sns.violinplot(y=df[column].dropna(), ax=ax, color=BLUE, inner="box", linewidth=1.0)
                ax.set_title(f"Violin Plot of {column}", fontsize=13, fontweight="bold", pad=10)
                out_path = os.path.join(OUTPUT_DIR, f"violin_{column}.png")
            _save(fig, out_path)

        # ── bar ───────────────────────────────────────────────────────────────
        elif chart_type == "bar":
            if not column:
                return {"error": "bar requires a 'column' argument"}
            counts = df[column].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(BG)
            _apply_style(ax)
            palette = sns.color_palette("Blues_r", len(counts))
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                        palette=palette, edgecolor="none")
            ax.set_title(f"Value Counts of {column}", fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            plt.xticks(rotation=40, ha="right")
            out_path = os.path.join(OUTPUT_DIR, f"bar_{column}.png")
            _save(fig, out_path)

        # ── heatmap ───────────────────────────────────────────────────────────
        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.empty:
                return {"error": "no numeric columns found for heatmap"}
            corr = numeric_df.corr()
            size = max(8, len(corr.columns) * 0.85)
            fig, ax = plt.subplots(figsize=(size, size * 0.8))
            fig.patch.set_facecolor(BG)
            _apply_style(ax)
            mask = None
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.4, linecolor="#30363d",
                annot_kws={"size": 9},
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
            plt.xticks(rotation=45, ha="right")
            out_path = os.path.join(OUTPUT_DIR, "heatmap_correlation.png")
            _save(fig, out_path)

        # ── scatter ───────────────────────────────────────────────────────────
        elif chart_type == "scatter":
            if not column or not column2:
                return {"error": "scatter requires both 'column' and 'column2' arguments"}
            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_facecolor(BG)
            _apply_style(ax)
            sns.scatterplot(x=df[column], y=df[column2], ax=ax, alpha=0.55,
                            color=BLUE, edgecolor="none", s=25)
            # Trend line
            valid = df[[column, column2]].dropna()
            if len(valid) > 3:
                sns.regplot(x=valid[column], y=valid[column2], ax=ax, scatter=False,
                            color=GREEN, line_kws={"linewidth": 1.5, "alpha": 0.8})
            ax.set_title(f"Scatter: {column}  vs  {column2}",
                         fontsize=13, fontweight="bold", pad=10)
            out_path = os.path.join(OUTPUT_DIR, f"scatter_{column}_vs_{column2}.png")
            _save(fig, out_path)

        # ── pairplot ──────────────────────────────────────────────────────────
        elif chart_type == "pairplot":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.empty:
                return {"error": "no numeric columns found for pairplot"}
            cols = numeric_df.columns[:5].tolist()
            g = sns.pairplot(
                df[cols].dropna(),
                plot_kws={"alpha": 0.45, "color": BLUE, "edgecolor": "none", "s": 12},
                diag_kws={"color": PURPLE, "alpha": 0.75, "edgecolor": "none"},
            )
            g.figure.patch.set_facecolor(BG)
            for ax in g.axes.flat:
                _apply_style(ax)
            out_path = os.path.join(OUTPUT_DIR, "pairplot.png")
            g.figure.savefig(out_path, bbox_inches="tight", dpi=100, facecolor=BG)
            plt.close(g.figure)

        else:
            return {
                "error": (
                    f"Unknown chart_type '{chart_type}'. "
                    "Supported: histogram, boxplot, violin, bar, heatmap, scatter, pairplot"
                )
            }

        return {"saved_to": out_path, "chart_type": chart_type, "column": column}

    except Exception as e:
        return {"error": str(e)}
