from dataclasses import asdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from .data import clean_dataframe, quick_sample
from .evaluation import evaluate
from .features import build_preprocessor, detect_feature_groups
from .models import default_model_names, get_models
from .reporting import save_model_bundle, save_results


def train_dataset(
    df,
    target: str,
    profile,
    project_name: str,
    model_names: list[str] | None = None,
    quick: bool = False,
    max_rows: int = 50_000,
    test_size: float = 0.20,
):
    X, y, dropped_columns = clean_dataframe(df, target)

    if y.nunique() < 2:
        raise ValueError("O target precisa de pelo menos duas classes.")

    if quick:
        X, y = quick_sample(X, y, max_rows=max_rows)

    groups = detect_feature_groups(X, profile)
    selected_models = model_names or default_model_names(has_text=bool(groups.text))

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(groups)
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    results = {}
    best_name = None
    best_model = None
    best_score = -1.0
    best_matrix = None
    best_report = None

    for name, model in get_models(selected_models).items():
        model.fit(X_train_ready, y_train)
        predictions = model.predict(X_test_ready)
        metrics, matrix, report = evaluate(y_test, predictions)
        results[name] = metrics.as_dict()

        if metrics.f1_macro > best_score:
            best_score = metrics.f1_macro
            best_name = name
            best_model = model
            best_matrix = matrix
            best_report = report

    model_path = Path("models") / f"{project_name}.joblib"
    result_dir = Path("results") / project_name

    bundle = {
        "project_name": project_name,
        "profile": profile.name,
        "target": target,
        "feature_columns": list(X.columns),
        "feature_groups": asdict(groups),
        "dropped_columns": dropped_columns,
        "preprocessor": preprocessor,
        "model": best_model,
        "model_name": best_name,
        "classes": [str(value) for value in sorted(y.unique(), key=str)],
    }
    save_model_bundle(model_path, bundle)

    summary = {
        "project_name": project_name,
        "profile": profile.name,
        "rows_used": len(X),
        "target": target,
        "classes": bundle["classes"],
        "feature_groups": bundle["feature_groups"],
        "dropped_columns": dropped_columns,
        "models": results,
        "best_model": best_name,
        "selection_metric": "f1_macro",
        "best_f1_macro": best_score,
        "model_file": str(model_path),
    }
    save_results(result_dir, summary, best_matrix, best_report)

    return summary
