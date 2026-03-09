import pandas as pd

def load_dataset(file_path: str) -> dict:
    """Load a CSV dataset and return basic preview information."""
    try:
        df = pd.read_csv(file_path)

        return {
            "file_path": file_path,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "first_5_rows": df.head(5).astype(str).to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}
