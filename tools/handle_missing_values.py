import json
import os
import pandas as pd
from langchain_core.tools import tool


@tool
def handle_missing_values(file_path: str, strategies: str) -> dict:
    """Apply missing value handling strategies to the dataset and save a cleaned version.

    The 'strategies' argument must be a JSON string mapping column names to strategies.
    Supported strategies: 'mean', 'median', 'mode', 'ffill', 'bfill', 'drop_row', 'drop_column', 'constant:VALUE'.

    Example strategies argument:
    '{"age": "mean", "salary": "median", "department": "mode", "notes": "drop_column"}'

    Returns a summary of what was applied and the path to the cleaned file.
    """
    try:
        df = pd.read_csv(file_path)
        strategy_map = json.loads(strategies)
        summary = []
        cols_to_drop = []

        for col, strategy in strategy_map.items():
            if col not in df.columns:
                summary.append({"column": col, "action": "skipped - column not found"})
                continue

            null_before = int(df[col].isna().sum())
            if null_before == 0:
                summary.append({"column": col, "action": "skipped - no missing values"})
                continue

            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
                summary.append({"column": col, "action": f"filled {null_before} nulls with mean ({round(df[col].mean(), 4)})"})
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
                summary.append({"column": col, "action": f"filled {null_before} nulls with median ({round(df[col].median(), 4)})"})
            elif strategy == "mode":
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                summary.append({"column": col, "action": f"filled {null_before} nulls with mode ('{mode_val}')"})
            elif strategy == "ffill":
                df[col] = df[col].ffill()
                summary.append({"column": col, "action": f"filled {null_before} nulls with forward fill"})
            elif strategy == "bfill":
                df[col] = df[col].bfill()
                summary.append({"column": col, "action": f"filled {null_before} nulls with backward fill"})
            elif strategy == "drop_row":
                df = df.dropna(subset=[col])
                summary.append({"column": col, "action": f"dropped rows with null {col} ({null_before} rows removed)"})
            elif strategy == "drop_column":
                cols_to_drop.append(col)
                summary.append({"column": col, "action": "column dropped entirely"})
            elif strategy.startswith("constant:"):
                constant_val = strategy.split(":", 1)[1]
                df[col] = df[col].fillna(constant_val)
                summary.append({"column": col, "action": f"filled {null_before} nulls with constant '{constant_val}'"})
            else:
                summary.append({"column": col, "action": f"unknown strategy '{strategy}' - skipped"})

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        base, ext = os.path.splitext(file_path)
        cleaned_path = f"{base}_cleaned{ext}"
        df.to_csv(cleaned_path, index=False)

        return {
            "cleaned_file_path": cleaned_path,
            "rows_remaining": len(df),
            "columns_remaining": len(df.columns),
            "remaining_nulls": int(df.isna().sum().sum()),
            "actions_applied": summary
        }
    except Exception as e:
        return {"error": str(e)}
