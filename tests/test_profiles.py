from src.profiles import detect_profile, resolve_target


def test_iot_profile():
    columns = ["flow_duration", "Rate", "label"]
    profile = detect_profile(columns)
    assert profile.name == "IoT Intrusion Detection"
    assert resolve_target(columns, profile) == "label"


def test_enron_profile():
    columns = ["Subject", "Body", "Label"]
    profile = detect_profile(columns)
    assert profile.name == "Enron Fraud Email"
    assert resolve_target(columns, profile) == "Label"


def test_malware_profile():
    columns = [*(f"F_{i}" for i in range(1, 40)), "label"]
    profile = detect_profile(columns)
    assert profile.name == "Malware Executable Detection"


def test_keylogger_profile():
    columns = ["Source Port", "Destination Port", "Class"]
    profile = detect_profile(columns)
    assert profile.name == "Keylogger Detection"
    assert resolve_target(columns, profile) == "Class"
