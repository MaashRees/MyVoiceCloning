from types import SimpleNamespace

import pytest

from voicestudio.config import Settings
from voicestudio.translate import TranslationError, Translator, language_name


class FakeClient:
    def __init__(self, content=None, error=None):
        self.calls = []
        self._content, self._error = content, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        msg = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make(client):
    return Translator(Settings(translate_model="m"), client=client)


def test_translate_builds_prompt_and_strips_quotes():
    client = FakeClient('"Hello world"')
    assert make(client).translate("Bonjour le monde", "fr", "en") == "Hello world"
    call = client.calls[0]
    assert "French to English" in call["messages"][0]["content"]
    assert call["messages"][1]["content"] == "Bonjour le monde"


def test_empty_text_makes_no_call():
    client = FakeClient("x")
    assert make(client).translate("   ") == ""
    assert client.calls == []


def test_errors_are_raised_not_returned_as_text():
    with pytest.raises(TranslationError):
        make(FakeClient(error=ConnectionError("boom"))).translate("salut")
    with pytest.raises(TranslationError):
        make(FakeClient("")).translate("salut")


def test_missing_configuration_is_explicit():
    with pytest.raises(TranslationError, match="TRANSLATE_API_KEY"):
        Translator(Settings(_env_file=None)).translate("salut")


def test_language_name():
    assert language_name("fr") == "French"
    assert language_name("Klingon") == "Klingon"
