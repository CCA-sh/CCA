import json
import pytest
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

from cca.sessions import (
    generate_session_id,
    _extract_display,
    save_session,
    load_session,
    mark_compacted_in_history,
    load_session_state,
)
from cca.llm.base import Message, TextContent, ToolUseContent, ToolResultContent

def get_history(sessions_dir):
    history_file = sessions_dir.parent / "history.jsonl"
    if not history_file.exists():
        return []
    res = []
    with open(history_file, "r") as f:
        for line in f:
            if line.strip():
                res.append(json.loads(line))
    return res

class TestGenerateSessionId:
    def test_standard(self):
        sid = generate_session_id()
        assert re.match(r"^\d{8}-\d{6}-[a-f0-9]{6}-comp0$", sid)
        
    def test_compaction_bump(self):
        sid = "20260503-104530-a1b2c3-comp0"
        sid2 = generate_session_id(is_compaction=True, current_id=sid)
        assert sid2 == "20260503-104530-a1b2c3-comp1"
        
        sid3 = generate_session_id(is_compaction=True, current_id=sid2)
        assert sid3 == "20260503-104530-a1b2c3-comp2"

    def test_compaction_bump_missing_comp(self):
        sid = "some-random-id"
        sid2 = generate_session_id(is_compaction=True, current_id=sid)
        assert sid2 == "some-random-id-comp1"

    def test_subagent(self):
        parent = "20260503-104530-a1b2c3-comp0"
        sid = generate_session_id(parent_id=parent)
        assert sid.startswith(parent + "-sub-")
        assert len(sid) == len(parent) + 5 + 6 # "-sub-" + 6 hex chars

class TestExtractDisplay:
    def test_string_content(self):
        msgs = [Message(role="user", content="Hello string!")]
        assert _extract_display(msgs) == "Hello string!"
        
    def test_text_blocks(self):
        msgs = [Message(role="user", content=[
            TextContent(text="Hello block!"),
            TextContent(text=" More text")
        ])]
        assert _extract_display(msgs) == "Hello block!"
        
    def test_mixed_blocks(self):
        msgs = [Message(role="user", content=[
            ToolResultContent(tool_use_id="1", content="ok", is_error=False),
            TextContent(text="This is what we want"),
        ])]
        assert _extract_display(msgs) == "This is what we want"

    def test_assistant_first_ignored(self):
        msgs = [
            Message(role="assistant", content="ignored"),
            Message(role="user", content="Hello user!"),
        ]
        assert _extract_display(msgs) == "Hello user!"
        
    def test_no_text(self):
        msgs = [Message(role="user", content=[
            ToolResultContent(tool_use_id="1", content="ok", is_error=False)
        ])]
        assert _extract_display(msgs) == "New Session"

@pytest.fixture()
def sessions_dir(tmp_path, monkeypatch):
    """Redirect SESSIONS_DIR and HISTORY_FILE to a temp dir."""
    import cca.sessions as sess_mod
    monkeypatch.setattr(sess_mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(sess_mod, "HISTORY_FILE", tmp_path / "history.jsonl")
    
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
    return tmp_path / "sessions"

class TestSessionPersistence:
    def _make_msgs(self, text="hello"):
        return [Message(role="user", content=[TextContent(text=text)])]

    def test_save_and_load(self, sessions_dir):
        msgs = self._make_msgs("hi there")
        sid = "20260503-104530-a1b2c3-comp0"
        save_session(sid, "/proj", msgs)
        
        loaded = load_session(sid)
        assert len(loaded) == 1
        assert loaded[0].content[0].text == "hi there"
        
        # Check history.jsonl
        history = get_history(sessions_dir)
        assert len(history) == 1
        assert history[0]["sessionId"] == sid
        assert history[0]["project"] == "/proj"
        assert history[0]["display"] == "hi there"

    def test_save_persists_safe_runtime_settings(self, sessions_dir):
        sid = "settings-test"
        settings = {
            "model": {"selector": "google:gemini-3.6-flash"},
            "thinking": {"requested": "auto", "effective": "medium"},
        }
        save_session(sid, "/proj", self._make_msgs(), settings=settings)

        payload = json.loads((sessions_dir / f"{sid}.json").read_text())
        assert payload["settings"] == settings

    def test_save_and_load_resumable_state(self, sessions_dir):
        sid = "state-test-comp2"
        state = {
            "trace_id": "trace",
            "agent_id": "root",
            "generation": 2,
            "next_call_idx": 7,
            "message_index_offset": 42,
            "compaction_summary": "surviving summary",
            "token_counts": {"input": 100, "output": 20},
        }
        save_session(sid, "/proj", self._make_msgs(), state=state)
        messages, restored = load_session_state(sid)
        assert messages[0].content[0].text == "hello"
        assert restored == state

    def test_save_overwrites_json_but_updates_history(self, sessions_dir):
        sid = "20260503-104530-a1b2c3-comp0"
        save_session(sid, "/proj", self._make_msgs("first"))
        save_session(sid, "/proj", self._make_msgs("second"))
        
        loaded = load_session(sid)
        assert loaded[0].content[0].text == "second"
        
        history = get_history(sessions_dir)
        # Should replace existing entry with the same sessionId, or at least keep 1
        assert len(history) == 1
        assert history[0]["display"] == "second"

    def test_mark_compacted(self, sessions_dir):
        sid = "comp-test-id"
        save_session(sid, "/proj", self._make_msgs("will be compacted"))
        
        mark_compacted_in_history(sid)
        history = get_history(sessions_dir)
        assert len(history) == 1
        assert history[0]["display"].startswith("[Compacted] will be compacted")
        
    def test_load_nonexistent_raises(self, sessions_dir):
        with pytest.raises(FileNotFoundError):
            load_session("doesnotexist")
