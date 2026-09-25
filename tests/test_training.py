from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction import predict_dataset
from src.profiles import detect_profile, resolve_target
from src.training import train_dataset


def _train(df, name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = detect_profile(list(df.columns))
    target = resolve_target(list(df.columns), profile)
    assert target is not None

    summary = train_dataset(
        df=df,
        target=target,
        profile=profile,
        project_name=name,
        model_names=["decision_tree"],
    )

    assert summary["best_model"] == "decision_tree"
    assert (tmp_path / "models" / f"{name}.joblib").exists()
    assert (tmp_path / "results" / name / "metrics.json").exists()
    return profile, target


def test_iot_dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "flow_duration": rng.random(n),
        "Rate": rng.normal(100, 20, n),
        "syn_count": rng.integers(0, 20, n),
        "label": np.where(np.arange(n) % 3 == 0, "DDoS", np.where(np.arange(n) % 3 == 1, "BenignTraffic", "Mirai")),
    })
    _train(df, "iot", tmp_path, monkeypatch)


def test_enron_text_dataset(tmp_path, monkeypatch):
    n = 120
    fraud = np.arange(n) % 2 == 0
    df = pd.DataFrame({
        "Subject": np.where(fraud, "urgent account payment", "team meeting update"),
        "Body": np.where(fraud, "verify account transfer money urgent click", "project meeting schedule report internal team"),
        "SenderType": np.where(fraud, "External", "Internal"),
        "Label": fraud.astype(int),
    })
    _train(df, "enron", tmp_path, monkeypatch)


def test_malware_dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    n = 120
    data = {f"F_{i}": rng.integers(0, 2, n) for i in range(1, 35)}
    data["label"] = np.where(np.arange(n) % 2 == 0, "malware", "benign")
    df = pd.DataFrame(data)
    _train(df, "malware", tmp_path, monkeypatch)


def test_keylogger_dataset_and_prediction(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame({
        "Duration": rng.random(n),
        "Protocol": np.where(np.arange(n) % 2 == 0, "TCP", "UDP"),
        "Bytes": rng.integers(10, 2000, n),
        "Class": np.where(np.arange(n) % 2 == 0, "Keylogger", "Benign"),
    })
    _train(df, "keylogger", tmp_path, monkeypatch)

    predict_input = df.drop(columns=["Class"]).head(10)
    input_path = tmp_path / "predict.csv"
    output_path = tmp_path / "predictions.csv"
    predict_input.to_csv(input_path, index=False)

    path, rows = predict_dataset(tmp_path / "models" / "keylogger.joblib", input_path, output_path)
    assert rows == 10
    assert Path(path).exists()
    output = pd.read_csv(path)
    assert "prediction" in output.columns
