import pandas as pd
from langchain_core.tools import tool


@tool
def get_statistics(file_path: str) -> dict:
    """Return statistics for numeric and categorical columns."""
    try:
        df = pd.read_csv(file_path)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        describe = df.describe(include="all").fillna("N/A").astype(str).to_dict()

        skewness = df[numeric_cols].skew().round(4).to_dict()
        kurtosis = df[numeric_cols].kurt().round(4).to_dict()

        categorical_summary = {}
        for col in categorical_cols:
            categorical_summary[col] = {
                "unique_values": int(df[col].nunique()),
                "top_5_values": df[col].value_counts().head(5).astype(int).to_dict()
            }

        return {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "describe": describe,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "categorical_summary": categorical_summary
        }

    except Exception as e:
        return {"error": str(e)}