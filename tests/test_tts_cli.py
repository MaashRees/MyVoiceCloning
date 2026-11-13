import asyncio

import pytest
from typer.testing import CliRunner

from voicestudio.cli import app
from voicestudio.tts import get_backend
from voicestudio.tts.edge import _run, pick_voice


def test_pick_voice():
    assert pick_voice("fr", "femme") == "fr-FR-VivienneMultilingualNeural"
    assert pick_voice("fr", None) == "fr-FR-RemyMultilingualNeural"
    assert pick_voice("fr", "fr-FR-DeniseNeural") == "fr-FR-DeniseNeural"
    with pytest.raises(ValueError):
        pick_voice("ja", "homme")


def test_run_works_with_and_without_running_loop():
    async def answer():
        return 42

    assert _run(answer()) == 42

    async def nested():
        return _run(answer())

    assert asyncio.run(nested()) == 42


def test_backends():
    assert get_backend("edge").supports_cloning is False
    chatterbox = get_backend("chatterbox")  # import paresseux : torch n'est pas requis ici
    assert chatterbox.supports_cloning and not chatterbox.sends_data_off_machine
    assert get_backend("remote").sends_data_off_machine
    with pytest.raises(ValueError):
        get_backend("nope")


def test_cli_help_and_unknown_backend():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("record-ref", "check-ref", "transcribe", "translate", "say", "devices"):
        assert command in result.output
    bad = runner.invoke(app, ["say", "salut", "--backend", "nope"])
    assert bad.exit_code == 1


def test_cli_check_ref(tmp_path):
    import numpy as np

    from voicestudio.audio import save_wav

    t = np.arange(16000 * 12) / 16000
    path = save_wav(tmp_path / "ref.wav", 0.2 * np.sin(2 * np.pi * 220 * t), 16000)
    result = CliRunner().invoke(app, ["check-ref", str(path)])
    assert result.exit_code == 0
    assert "Qualité correcte" in result.output
