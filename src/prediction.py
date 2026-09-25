from pathlib import Path

import joblib
import pandas as pd

from .data import load_dataset


def predict_dataset(model_path: str | Path, dataset_path: str | Path, output_path: str | Path):
    bundle = joblib.load(model_path)
    df = load_dataset(dataset_path)

    required = bundle["feature_columns"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Faltam colunas no dataset: " + ", ".join(missing[:20]))

    X = df[required]
    X_ready = bundle["preprocessor"].transform(X)
    predictions = bundle["model"].predict(X_ready)

    output = df.copy()
    output["prediction"] = predictions

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path, len(output)
