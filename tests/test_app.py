from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / 'app.py'


def test_demo_build_chat_and_reset():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state['index'] is not None
    at.chat_input[0].set_value('事务的 ACID 特性分别是什么？').run()
    assert not at.exception
    messages = at.session_state['messages']
    assert len(messages) == 2
    assert '原子性' in messages[-1]['text']
    assert any(s.metadata['page'] == 2 for s in messages[-1]['sources'])
    at.radio[0].set_value('OpenAI 问答').run()
    assert not at.exception
    assert at.session_state['index'] is None
    assert at.session_state['messages'] == []
    assert at.chat_input[0].disabled


def test_openai_requires_key_and_consent(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.radio[0].set_value('OpenAI 问答').run()
    at.button[0].click().run()
    assert at.warning
    assert at.session_state['index'] is None
    assert not at.exception
