from __future__ import annotations

from pathlib import Path

import pytest

from backend.server import server_utils
from gpt_researcher.utils.report_continuation import IncompleteReportError


class Socket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


class Manager:
    async def start_streaming(self, *args, **kwargs):
        raise IncompleteReportError(
            "# partial report", finish_reason="length", continuations=2
        )


@pytest.mark.asyncio
async def test_websocket_boundary_writes_only_partial_markdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    socket = Socket()
    data = (
        'start {"task":"topic","report_type":"research_report",'
        '"report_source":"web","source_urls":[],"document_urls":[],'
        '"tone":"Objective"}'
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError("normal artifact generation must not run")

    monkeypatch.setattr(server_utils, "generate_report_files", forbidden)
    await server_utils.handle_start_command(socket, data, Manager())

    partials = list((tmp_path / "outputs").glob("*.partial.md"))
    assert len(partials) == 1
    assert partials[0].read_text() == "# partial report"
    assert not list((tmp_path / "outputs").glob("*.pdf"))
    assert not list((tmp_path / "outputs").glob("*.docx"))
    assert socket.messages[-1]["type"] == "incomplete_report"
    assert socket.messages[-1]["partial_path"].endswith(".partial.md")
