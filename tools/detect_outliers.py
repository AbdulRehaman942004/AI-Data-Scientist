import numpy as np
import pandas as pd
from langchain_core.tools import tool


@tool
def detect_outliers(file_path: str) -> dict:
    """Detect outliers in all numeric columns using IQR and Z-score methods.

    Call this after loading the dataset to quantify data quality issues.
    Returns outlier counts, percentages, bounds, and severity ratings per column.
    Severity: 'high' if IQR outlier % > 10%, 'medium' if > 5%, else 'low'.
    """
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return {"error": "No numeric columns found in the dataset."}

        outlier_report = []
        for col in numeric_cols:
            series = df[col].dropna()
            n = len(series)
            if n < 4:
                continue

            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            iqr_outliers = int(((series < lower) | (series > upper)).sum())
            iqr_pct = round(iqr_outliers / n * 100, 2)

            # Z-score method
            if series.std() > 0:
                z_scores = np.abs((series - series.mean()) / series.std())
                zscore_outliers = int((z_scores > 3).sum())
                zscore_pct = round(zscore_outliers / n * 100, 2)
            else:
                zscore_outliers = 0
                zscore_pct = 0.0

            severity = "high" if iqr_pct > 10 else "medium" if iqr_pct > 5 else "low"

            outlier_report.append({
                "column": col,
                "iqr_outliers": iqr_outliers,
                "iqr_outlier_pct": iqr_pct,
                "zscore_outliers": zscore_outliers,
                "zscore_outlier_pct": zscore_pct,
                "iqr_lower_bound": round(float(lower), 4),
                "iqr_upper_bound": round(float(upper), 4),
                "actual_min": round(float(series.min()), 4),
                "actual_max": round(float(series.max()), 4),
                "severity": severity,
            })

        return {
            "outlier_summary": outlier_report,
            "columns_with_high_outliers": [
                r["column"] for r in outlier_report if r["severity"] == "high"
            ],
            "columns_with_medium_outliers": [
                r["column"] for r in outlier_report if r["severity"] == "medium"
            ],
        }

    except Exception as e:
        return {"error": str(e)}
