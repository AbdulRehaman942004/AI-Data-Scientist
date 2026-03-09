import pandas as pd
from langchain_core.tools import tool


@tool
def get_dataframe_info(file_path: str) -> dict:
    """Get structural information about the dataset: column names, data types,
    non-null counts, and memory usage. Call this first to understand the dataset structure."""
    try:
        df = pd.read_csv(file_path)
        columns_info = []
        for col in df.columns:
            columns_info.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "total_count": len(df)
            })
        return {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns_info": columns_info,
            "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
            "column_names": df.columns.tolist()
        }
    except Exception as e:
        return {"error": str(e)}
