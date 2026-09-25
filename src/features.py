from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .profiles import DatasetProfile


@dataclass(frozen=True)
class FeatureGroups:
    numeric: list[str]
    categorical: list[str]
    text: list[str]


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def detect_feature_groups(X: pd.DataFrame, profile: DatasetProfile) -> FeatureGroups:
    numeric = list(X.select_dtypes(include="number").columns)
    non_numeric = [c for c in X.columns if c not in numeric]

    explicit_text = {_norm(c) for c in profile.text_columns}
    text: list[str] = []
    categorical: list[str] = []

    for column in non_numeric:
        series = X[column].dropna().astype(str)
        avg_length = float(series.str.len().mean()) if not series.empty else 0.0
        unique_count = int(series.nunique())
        unique_ratio = unique_count / max(len(series), 1)

        looks_like_text = (
            _norm(column) in explicit_text
            or avg_length >= 35
            or (unique_count > 100 and unique_ratio > 0.30)
        )

        if looks_like_text:
            text.append(column)
        else:
            categorical.append(column)

    return FeatureGroups(numeric=numeric, categorical=categorical, text=text)


def _prepare_text(values):
    if hasattr(values, "fillna"):
        return values.fillna("").astype(str)
    return pd.Series(values).fillna("").astype(str)


def build_preprocessor(groups: FeatureGroups) -> ColumnTransformer:
    transformers = []

    if groups.numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ])
        transformers.append(("numeric", numeric_pipeline, groups.numeric))

    if groups.categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ])
        transformers.append(("categorical", categorical_pipeline, groups.categorical))

    for index, column in enumerate(groups.text):
        text_pipeline = Pipeline([
            ("prepare", FunctionTransformer(_prepare_text, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ])
        transformers.append((f"text_{index}", text_pipeline, column))

    if not transformers:
        raise ValueError("Não foram encontradas features utilizáveis.")

    return ColumnTransformer(transformers, remainder="drop")
