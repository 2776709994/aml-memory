import os

os.environ.setdefault("AML_SKIP_DOTENV", "1")
os.environ.setdefault("EMBED_PROVIDER", "hash")
os.environ.setdefault("AML_DISABLE_AUTH", "1")

from aml_memory.config import load_settings
from aml_memory.ingest import Ingestor, Message
from aml_memory.llm import LLM


def _units(mode, monkeypatch):
    monkeypatch.setenv("AML_EMBED_TEXT_MODE", mode)
    settings = load_settings()
    ingestor = Ingestor(settings, LLM(settings))
    units = ingestor.build_units(
        request_id=f"req-{mode}",
        user_id="u-embed",
        session_id="s1",
        messages=[
            Message(role="user", content="Caroline: I signed up for a pottery class yesterday.",
                    timestamp=1683388800000),
        ],
    )
    return units


def test_header_mode_embeds_the_stored_record(monkeypatch):
    unit = _units("header", monkeypatch)[0]
    assert unit.embed_text is None
    assert "pottery" in unit.text
    assert "6 May 2023" in unit.text  # the reader-facing record is dated


def test_speaker_mode_drops_the_date_but_keeps_the_name(monkeypatch):
    unit = _units("speaker", monkeypatch)[0]
    assert unit.embed_text is not None
    assert unit.embed_text.startswith("Caroline: ")
    assert "(6 May 2023)" not in unit.embed_text
    # the relative expression is still anchored inside the body
    assert "yesterday (5 May 2023)" in unit.embed_text
    assert "6 May 2023" in unit.text


def test_body_mode_drops_speaker_and_date(monkeypatch):
    unit = _units("body", monkeypatch)[0]
    assert unit.embed_text is not None
    assert not unit.embed_text.startswith("Caroline: ")
    assert "(6 May 2023)" not in unit.embed_text
    assert "pottery" in unit.embed_text
