import pytest

from scripts.eval_financial50_live import bind_experiment


def test_report_identity_fences_legacy_and_changed_code():
    report = {}
    key = bind_experiment(report, "questions", "code-v1")
    assert bind_experiment(report, "questions", "code-v1") == key
    assert bind_experiment({}, "questions", "code-v2") != key
    with pytest.raises(ValueError):
        bind_experiment(report, "questions", "code-v2")
    with pytest.raises(ValueError):
        bind_experiment({"cases": []}, "questions", "code-v1")
