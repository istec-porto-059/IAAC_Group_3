from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


ALL_MODELS = {
    "decision_tree": lambda: DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=80,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ),
    "logistic_regression": lambda: LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    ),
    "linear_svm": lambda: LinearSVC(
        class_weight="balanced",
        random_state=42,
    ),
    "sgd": lambda: SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    ),
}


def default_model_names(has_text: bool) -> list[str]:
    if has_text:
        return ["logistic_regression", "linear_svm", "sgd"]
    return ["decision_tree", "random_forest", "logistic_regression"]


def get_models(names: list[str]):
    unknown = [name for name in names if name not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Modelos desconhecidos: {', '.join(unknown)}")
    return {name: ALL_MODELS[name]() for name in names}
