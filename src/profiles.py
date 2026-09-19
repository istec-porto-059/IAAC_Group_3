from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    target_candidates: tuple[str, ...]
    text_columns: tuple[str, ...] = ()
    mode: str = "tabular"


PROFILES = {
    "iot": DatasetProfile(
        name="IoT Intrusion Detection",
        target_candidates=("label", "Label"),
    ),
    "enron": DatasetProfile(
        name="Enron Fraud Email",
        target_candidates=("Label", "label", "Fraud", "fraud", "is_fraud"),
        text_columns=("Subject", "subject", "Body", "body", "Message", "message", "text", "Email Text"),
        mode="text",
    ),
    "malware": DatasetProfile(
        name="Malware Executable Detection",
        target_candidates=("label", "Label", "malware", "Malware", "legitimate"),
    ),
    "keylogger": DatasetProfile(
        name="Keylogger Detection",
        target_candidates=("Class", "class", "label", "Label"),
    ),
    "generic": DatasetProfile(
        name="Generic Classification Dataset",
        target_candidates=("label", "Label", "class", "Class", "target", "Target", "category", "Category"),
    ),
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def detect_profile(columns: list[str]) -> DatasetProfile:
    normalized = {_norm(c): c for c in columns}

    if "flowduration" in normalized and "label" in normalized:
        return PROFILES["iot"]

    has_subject = any(key in normalized for key in ("subject", "emailsubject"))
    has_body = any(key in normalized for key in ("body", "message", "emailtext", "text"))
    if has_subject or has_body:
        return PROFILES["enron"]

    f_features = sum(bool(re.fullmatch(r"f\d+", key)) for key in normalized)
    if f_features >= 20 and "label" in normalized:
        return PROFILES["malware"]

    if "class" in normalized:
        return PROFILES["keylogger"]

    return PROFILES["generic"]


def resolve_target(columns: list[str], profile: DatasetProfile, requested: str | None = None) -> str | None:
    exact = {str(c): str(c) for c in columns}
    normalized = {_norm(c): str(c) for c in columns}

    if requested:
        if requested in exact:
            return requested
        return normalized.get(_norm(requested))

    candidates = profile.target_candidates + PROFILES["generic"].target_candidates
    for candidate in candidates:
        match = normalized.get(_norm(candidate))
        if match:
            return match
    return None
