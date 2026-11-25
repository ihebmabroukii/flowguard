import pandas as pd
import pytest

from flowguard.data.labels import UnknownLabelError, is_attempted_label, normalize_label


@pytest.mark.parametrize(
    ("raw", "family"),
    [
        # improved CIC-IDS2017
        ("BENIGN", "Benign"),
        ("DoS Hulk", "DoS"),
        ("DoS GoldenEye", "DoS"),
        ("DoS Slowloris", "DoS"),
        ("DoS Slowhttptest", "DoS"),
        ("DDoS", "DDoS"),
        ("Portscan", "PortScan"),
        ("PortScan", "PortScan"),
        ("FTP-Patator", "BruteForce"),
        ("SSH-Patator", "BruteForce"),
        ("Web Attack - Brute Force", "WebAttack"),
        ("Web Attack \x96 Brute Force", "WebAttack"),
        ("Web Attack - XSS", "WebAttack"),
        ("Web Attack - SQL Injection", "WebAttack"),
        ("Botnet", "Botnet"),
        ("Bot", "Botnet"),
        ("Infiltration", "Infiltration"),
        ("Infiltration - Portscan", "PortScan"),
        ("Heartbleed", "drop"),
        # CSE-CIC-IDS2018 (original and improved spellings)
        ("Benign", "Benign"),
        ("DoS attacks-Hulk", "DoS"),
        ("DoS attacks-SlowHTTPTest", "DoS"),
        ("DDoS attacks-LOIC-HTTP", "DDoS"),
        ("DDOS attack-HOIC", "DDoS"),
        ("DDOS attack-LOIC-UDP", "DDoS"),
        ("DDoS-LOIC-HTTP", "DDoS"),
        ("FTP-BruteForce", "BruteForce"),
        ("SSH-Bruteforce", "BruteForce"),
        ("Brute Force -Web", "WebAttack"),
        ("Brute Force -XSS", "WebAttack"),
        ("SQL Injection", "WebAttack"),
        ("Infilteration", "Infiltration"),
        ("Infiltration - NMAP Portscan", "PortScan"),
        ("Infiltration - Dropbox Download", "Infiltration"),
        ("Botnet Ares", "Botnet"),
        ("Label", "drop"),
    ],
)
def test_family_of_known_labels(mapper, raw, family):
    assert mapper.family_of(raw) == family


def test_normalize_label():
    assert normalize_label("  Web Attack \x96 Brute Force - Attempted ") == "web attack brute force"
    assert normalize_label("DoS attacks-SlowHTTPTest") == "dos attacks slowhttptest"


def test_attempted_detection():
    assert is_attempted_label("DoS Hulk - Attempted")
    assert not is_attempted_label("DoS Hulk")


def test_unknown_labels_are_all_reported(mapper):
    with pytest.raises(UnknownLabelError) as err:
        mapper.lookup(["BENIGN", "Quantum Teleport", "Mystery"])
    assert "Quantum Teleport" in str(err.value) and "Mystery" in str(err.value)


def test_attempted_flows_become_benign(mapper):
    labels = pd.Series(["BENIGN", "DoS Hulk", "DoS Hulk - Attempted", "Portscan", "Heartbleed"])
    category = pd.Series([-1, -1, 2, 3, -1])  # the Portscan row is attempted via the column only
    assert mapper.map_series(labels, category).tolist() == [
        "Benign", "DoS", "Benign", "Benign", "drop"
    ]


def test_attempted_policy_drop(mapper):
    drop_mapper = type(mapper)(mapper.rules, mapper.families, "drop")
    out = drop_mapper.map_series(pd.Series(["DDoS - Attempted", "DDoS"]))
    assert out.tolist() == ["drop", "DDoS"]


def test_every_rule_targets_a_known_family(mapper):
    for _, family in mapper.rules:
        assert family == "drop" or family in mapper.families
