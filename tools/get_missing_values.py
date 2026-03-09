import pandas as pd
from langchain_core.tools import tool


@tool
def get_missing_values(file_path: str) -> dict:
    """Analyze missing values in the dataset."""
    try:
        df = pd.read_csv(file_path)

        total_rows = len(df)
        null_counts = df.isna().sum()
        null_pct = ((null_counts / total_rows) * 100).round(2)

        missing_report = []
        for col in df.columns:
            missing_report.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "null_count": int(null_counts[col]),
                "null_percentage": float(null_pct[col]),
                "high_concern": null_pct[col] > 30
            })

        return {
            "total_rows": total_rows,
            "total_columns": df.shape[1],
            "columns_with_missing": [r for r in missing_report if r["null_count"] > 0],
            "high_concern_columns": [r["column"] for r in missing_report if r["high_concern"]],
            "total_missing_cells": int(null_counts.sum()),
            "dataset_completeness_pct": round((1 - null_counts.sum() / df.size) * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}