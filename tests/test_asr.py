from voicestudio.asr import Segment, load_vocab_prompt, to_srt
from voicestudio.asr.srt import format_timestamp
from voicestudio.asr.whisper import resolve_device


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(-1) == "00:00:00,000"


def test_to_srt():
    srt = to_srt([Segment(0, 1.5, "Bonjour"), Segment(1.5, 3, "le monde")])
    assert srt == (
        "1\n00:00:00,000 --> 00:00:01,500\nBonjour\n\n2\n00:00:01,500 --> 00:00:03,000\nle monde\n"
    )


def test_vocab_prompt(tmp_path):
    f = tmp_path / "vocab.txt"
    f.write_text("# commentaire\nLisbonne, Algarve\n\nRéunion\n", encoding="utf-8")
    assert load_vocab_prompt(f) == "Vocabulaire : Lisbonne, Algarve, Réunion."


def test_vocab_prompt_missing_or_empty(tmp_path):
    assert load_vocab_prompt(None) is None
    assert load_vocab_prompt(tmp_path / "absent.txt") is None
    empty = tmp_path / "e.txt"
    empty.write_text("# rien\n", encoding="utf-8")
    assert load_vocab_prompt(empty) is None


def test_vocab_prompt_is_capped(tmp_path):
    f = tmp_path / "v.txt"
    f.write_text(", ".join(f"mot{i}" for i in range(500)), encoding="utf-8")
    assert len(load_vocab_prompt(f)) <= 600


def test_resolve_device_explicit():
    assert resolve_device("cpu", "auto") == ("cpu", "int8")
    assert resolve_device("cuda", "auto") == ("cuda", "float16")
    assert resolve_device("cpu", "float32") == ("cpu", "float32")
