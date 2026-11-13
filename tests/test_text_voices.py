import pytest

from voicestudio.config import Settings
from voicestudio.tts.text import split_text
from voicestudio.voices import list_voices, resolve_reference


def test_split_text_groups_short_sentences():
    text = "Bonjour. Ça va ? Très bien !"
    assert split_text(text, max_chars=250) == ["Bonjour. Ça va ? Très bien !"]


def test_split_text_respects_limit_and_keeps_all_words():
    sentence = "Voici une phrase assez longue, avec des virgules, " * 12
    chunks = split_text(sentence, max_chars=80)
    assert all(len(c) <= 80 for c in chunks)
    assert " ".join(chunks).split() == sentence.split()


def test_split_text_splits_on_sentence_boundaries_first():
    text = "A" * 100 + ". " + "B" * 100 + "."
    assert split_text(text, max_chars=120) == ["A" * 100 + ".", "B" * 100 + "."]


def test_split_text_handles_unbreakable_word_and_empty():
    assert all(len(c) <= 10 for c in split_text("x" * 35, max_chars=10))
    assert split_text("   \n ") == []


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


def test_voice_profiles(settings):
    assert list_voices(settings) == {}
    with pytest.raises(FileNotFoundError, match="record-ref moi"):
        resolve_reference("moi", settings=settings)

    d = settings.voices_dir / "moi"
    d.mkdir(parents=True)
    (d / "ref_energique.wav").write_bytes(b"x")
    assert resolve_reference("moi", settings=settings).name == "ref_energique.wav"  # seul choix
    (d / "ref_neutre.wav").write_bytes(b"x")
    assert resolve_reference("moi", settings=settings).name == "ref_neutre.wav"  # défaut
    assert resolve_reference("moi", "energique", settings).name == "ref_energique.wav"
    assert list_voices(settings) == {"moi": ["energique", "neutre"]}
    with pytest.raises(FileNotFoundError):
        resolve_reference("moi", "triste", settings)
