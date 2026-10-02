from pathlib import Path
import json

import joblib


def save_model_bundle(path: str | Path, bundle: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def save_results(output_dir: str | Path, summary: dict, confusion_matrix, classification_report) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)

    confusion_matrix.to_csv(output_dir / "confusion_matrix.csv")
    classification_report.to_csv(output_dir / "classification_report.csv")
    return output_dir
