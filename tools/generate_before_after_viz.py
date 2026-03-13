import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langchain_core.tools import tool

OUTPUT_DIR = "outputs"


@tool
def generate_before_after_plots(original_file_path: str, cleaned_file_path: str) -> dict:
    """Generate side-by-side before/after comparison visualizations for data cleaning impact.

    ALWAYS call this immediately after handle_missing_values to visualize the cleaning impact.

    Args:
        original_file_path: Path to the original (uncleaned) CSV file.
        cleaned_file_path: The cleaned_file_path returned by handle_missing_values.

    Creates:
      - Side-by-side distribution histograms for each modified numeric column
      - A grouped bar chart comparing missing value counts before vs after

    All plots are saved with 'comparison_' prefix to outputs/.
    Returns before/after statistics and list of saved plot paths.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df_before = pd.read_csv(original_file_path)
        df_after = pd.read_csv(cleaned_file_path)

        before_stats = {
            "rows": int(len(df_before)),
            "columns": int(len(df_before.columns)),
            "missing_cells": int(df_before.isna().sum().sum()),
            "missing_pct": round(
                df_before.isna().sum().sum() / max(df_before.size, 1) * 100, 2
            ),
        }
        after_stats = {
            "rows": int(len(df_after)),
            "columns": int(len(df_after.columns)),
            "missing_cells": int(df_after.isna().sum().sum()),
            "missing_pct": round(
                df_after.isna().sum().sum() / max(df_after.size, 1) * 100, 2
            ),
        }

        numeric_before = set(df_before.select_dtypes(include="number").columns)
        numeric_after = set(df_after.select_dtypes(include="number").columns)
        common_numeric = list(numeric_before & numeric_after)

        saved_plots = []

        # --- Per-column distribution comparison ---
        changed_cols = [
            c for c in common_numeric
            if df_before[c].isna().sum() > 0
        ]

        BLUE = "#1f6feb"
        GREEN = "#3fb950"
        RED = "#f85149"
        BG = "#1c2128"
        PANEL = "#0d1117"
        MUTED = "#8b949e"
        TEXT = "#e6edf3"
        BORDER = "#30363d"

        for col in changed_cols[:6]:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            fig.patch.set_facecolor(BG)
            fig.suptitle(
                f"Before vs After Cleaning  ·  {col}",
                fontsize=13, fontweight="bold", color=TEXT, y=1.02,
            )

            for ax in axes:
                ax.set_facecolor(PANEL)
                ax.tick_params(colors=MUTED)
                ax.xaxis.label.set_color(MUTED)
                ax.yaxis.label.set_color(MUTED)
                for spine in ax.spines.values():
                    spine.set_edgecolor(BORDER)

            n_miss_before = int(df_before[col].isna().sum())
            n_miss_after = int(df_after[col].isna().sum())

            sns.histplot(
                df_before[col].dropna(), kde=True, ax=axes[0],
                color=BLUE, alpha=0.75, edgecolor="none",
            )
            axes[0].set_title(
                f"BEFORE  ·  {n_miss_before} missing values",
                color=RED, fontsize=11, pad=8,
            )
            axes[0].set_xlabel(col, color=MUTED)

            sns.histplot(
                df_after[col].dropna(), kde=True, ax=axes[1],
                color=GREEN, alpha=0.75, edgecolor="none",
            )
            axes[1].set_title(
                f"AFTER  ·  {n_miss_after} missing values",
                color=GREEN, fontsize=11, pad=8,
            )
            axes[1].set_xlabel(col, color=MUTED)

            plt.tight_layout()
            out_path = os.path.join(OUTPUT_DIR, f"comparison_{col}.png")
            fig.savefig(out_path, bbox_inches="tight", dpi=100, facecolor=BG)
            plt.close(fig)
            saved_plots.append(out_path)

        # --- Missing values grouped bar chart ---
        all_missing_before = df_before.isna().sum()
        cols_with_missing = all_missing_before[all_missing_before > 0].index.tolist()

        if cols_with_missing:
            fig_w = max(8, len(cols_with_missing) * 1.3)
            fig, ax = plt.subplots(figsize=(fig_w, 5))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(PANEL)

            x = range(len(cols_with_missing))
            width = 0.38

            before_counts = [int(df_before[c].isna().sum()) for c in cols_with_missing]
            after_counts = [
                int(df_after[c].isna().sum()) if c in df_after.columns else 0
                for c in cols_with_missing
            ]

            ax.bar(
                [i - width / 2 for i in x], before_counts, width,
                label="Before", color=RED, alpha=0.82,
            )
            ax.bar(
                [i + width / 2 for i in x], after_counts, width,
                label="After", color=GREEN, alpha=0.82,
            )

            ax.set_xlabel("Columns", color=MUTED, fontsize=11)
            ax.set_ylabel("Missing Value Count", color=MUTED, fontsize=11)
            ax.set_title(
                "Missing Values: Before vs After Cleaning",
                color=TEXT, fontsize=13, fontweight="bold",
            )
            ax.set_xticks(list(x))
            ax.set_xticklabels(cols_with_missing, rotation=45, ha="right", color=MUTED)
            ax.tick_params(colors=MUTED)
            ax.legend(
                facecolor=BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=10,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

            plt.tight_layout()
            out_path = os.path.join(OUTPUT_DIR, "comparison_missing_values_summary.png")
            fig.savefig(out_path, bbox_inches="tight", dpi=100, facecolor=BG)
            plt.close(fig)
            saved_plots.append(out_path)

        return {
            "before_stats": before_stats,
            "after_stats": after_stats,
            "rows_removed": before_stats["rows"] - after_stats["rows"],
            "columns_removed": before_stats["columns"] - after_stats["columns"],
            "missing_cells_fixed": before_stats["missing_cells"] - after_stats["missing_cells"],
            "comparison_plots_saved": saved_plots,
        }

    except Exception as e:
        return {"error": str(e)}
