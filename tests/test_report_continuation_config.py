from __future__ import annotations

import pytest

from gpt_researcher.config.config import Config


def test_report_continuation_defaults():
    config = Config("default")
    assert config.report_max_continuations == 2
    assert not hasattr(config, "report_continuation_tail_chars")


@pytest.mark.parametrize("value", ["-1", "11"])
def test_report_max_continuations_is_bounded(monkeypatch, value):
    monkeypatch.setenv("REPORT_MAX_CONTINUATIONS", value)
    with pytest.raises(ValueError, match="between 0 and 10"):
        Config("default")
