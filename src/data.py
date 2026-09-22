from pathlib import Path
import csv

import numpy as np
import pandas as pd


ID_LIKE_NAMES = {
    "id", "index", "filename", "file", "filepath", "path", "messageid",
    "md5", "sha1", "sha256", "hash",
}


def _normalized(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _detect_separator(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            sample = handle.read(8192)
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except (csv.Error, OSError):
        return ","


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        return pd.read_csv(path, sep=_detect_separator(path), low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError("Formato não suportado. Usa CSV, TSV, TXT ou Parquet.")


def clean_dataframe(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if target not in df.columns:
        raise ValueError(f"A coluna target '{target}' não existe no dataset.")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target])

    y = df[target]
    X = df.drop(columns=[target])

    dropped: list[str] = []
    for column in list(X.columns):
        normalized = _normalized(column)
        series = X[column]

        if normalized.startswith("unnamed") or series.isna().all() or series.nunique(dropna=True) <= 1:
            dropped.append(column)
            continue

        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        if normalized in ID_LIKE_NAMES and unique_ratio > 0.80:
            dropped.append(column)

    if dropped:
        X = X.drop(columns=dropped)

    if X.empty:
        raise ValueError("Não sobraram colunas para treinar o modelo.")

    return X, y, dropped


def quick_sample(X: pd.DataFrame, y: pd.Series, max_rows: int, random_state: int = 42):
    if len(X) <= max_rows:
        return X, y

    combined = X.copy()
    combined["__target__"] = y.to_numpy()

    try:
        sampled, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
            combined,
            train_size=max_rows,
            stratify=combined["__target__"],
            random_state=random_state,
        )
    except ValueError:
        sampled = combined.sample(n=max_rows, random_state=random_state)

    return sampled.drop(columns=["__target__"]), sampled["__target__"]
