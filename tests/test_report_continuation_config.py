from __future__ import annotations

import pytest

from gpt_researcher.config.config import Config


def test_report_continuation_defaults():
    config = Config("default")
    assert config.report_max_continuations == 2
    assert config.report_continuation_tail_chars == 65_536


@pytest.mark.parametrize("value", ["-1", "11"])
def test_report_max_continuations_is_bounded(monkeypatch, value):
    monkeypatch.setenv("REPORT_MAX_CONTINUATIONS", value)
    with pytest.raises(ValueError, match="between 0 and 10"):
        Config("default")


def test_report_continuation_tail_must_be_positive(monkeypatch):
    monkeypatch.setenv("REPORT_CONTINUATION_TAIL_CHARS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        Config("default")
