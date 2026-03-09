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


@tool
def generate_visualization(file_path: str, chart_type: str, column: str = "", column2: str = "") -> dict:
    """Generate and save a visualization for the dataset.

    Supported chart_type values:
    - 'histogram': Distribution of a numeric column. Requires 'column'.
    - 'boxplot': Box plot of a numeric column to show outliers. Requires 'column'.
    - 'bar': Value counts of a categorical column. Requires 'column'.
    - 'heatmap': Correlation matrix of all numeric columns. No column needed.
    - 'scatter': Scatter plot between two columns. Requires 'column' and 'column2'.
    - 'pairplot': Pairwise relationships for all numeric columns. No column needed.

    Returns the file path of the saved plot.
    """
    try:
        df = pd.read_csv(file_path)
        _ensure_output_dir()
        sns.set_theme(style="whitegrid")

        if chart_type == "histogram":
            if not column:
                return {"error": "histogram requires a 'column' argument"}
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[column].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            out_path = os.path.join(OUTPUT_DIR, f"histogram_{column}.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        elif chart_type == "boxplot":
            if not column:
                return {"error": "boxplot requires a 'column' argument"}
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.boxplot(y=df[column].dropna(), ax=ax)
            ax.set_title(f"Box Plot of {column}")
            out_path = os.path.join(OUTPUT_DIR, f"boxplot_{column}.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        elif chart_type == "bar":
            if not column:
                return {"error": "bar requires a 'column' argument"}
            counts = df[column].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
            ax.set_title(f"Value Counts of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            out_path = os.path.join(OUTPUT_DIR, f"bar_{column}.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.empty:
                return {"error": "no numeric columns found for heatmap"}
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            out_path = os.path.join(OUTPUT_DIR, "heatmap_correlation.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        elif chart_type == "scatter":
            if not column or not column2:
                return {"error": "scatter requires both 'column' and 'column2' arguments"}
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=df[column], y=df[column2], ax=ax, alpha=0.6)
            ax.set_title(f"Scatter: {column} vs {column2}")
            out_path = os.path.join(OUTPUT_DIR, f"scatter_{column}_vs_{column2}.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        elif chart_type == "pairplot":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.empty:
                return {"error": "no numeric columns found for pairplot"}
            cols = numeric_df.columns[:5].tolist()
            fig = sns.pairplot(df[cols].dropna()).fig
            out_path = os.path.join(OUTPUT_DIR, "pairplot.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        else:
            return {"error": f"unknown chart_type '{chart_type}'. Supported: histogram, boxplot, bar, heatmap, scatter, pairplot"}

        return {"saved_to": out_path, "chart_type": chart_type, "column": column}

    except Exception as e:
        return {"error": str(e)}
