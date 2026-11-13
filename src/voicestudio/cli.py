"""Ligne de commande : ``voicestudio --help``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from voicestudio.config import get_settings

app = typer.Typer(help="Voix off personnalisée : enregistrer, transcrire, traduire, synthétiser.")


def _fail(message: str) -> typer.Exit:
    typer.secho(f"Erreur : {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


@app.command()
def devices() -> None:
    """Liste les micros disponibles."""
    from voicestudio.audio.record import list_input_devices

    try:
        found = list_input_devices()
    except RuntimeError as exc:
        raise _fail(str(exc)) from exc
    if not found:
        raise _fail("Aucun micro détecté.")
    for d in found:
        typer.echo(f"[{d.index}] {d.name}  ({d.default_samplerate} Hz)")


@app.command("record-ref")
def record_ref(
    name: str = typer.Argument(..., help="Nom du profil de voix, ex. « moi »."),
    label: str = typer.Option("neutre", help="Ambiance de la prise : neutre, energique, calme..."),
    duration: float | None = typer.Option(
        None, "--duration", "-d", help="Durée en secondes (sinon : jusqu'à Entrée)."
    ),
    device: int | None = typer.Option(None, "--device", help="Numéro du micro (voir `devices`)."),
) -> None:
    """Enregistre un extrait de référence et vérifie sa qualité."""
    from voicestudio.audio import analyze, reference_warnings, save_wav, trim_silence
    from voicestudio.audio.processing import to_float
    from voicestudio.audio.record import record, stop_on_enter

    typer.echo(
        "Conseils : pièce calme, micro à ~15-20 cm, ton naturel, 10 à 30 s de phrases continues."
    )
    typer.echo("Enregistrement en cours... " + ("" if duration else "(Entrée pour arrêter)"))
    stop = None if duration else stop_on_enter()
    try:
        audio, sr = record(duration_s=duration, device=device, stop_event=stop)
    except RuntimeError as exc:
        raise _fail(str(exc)) from exc
    if len(audio) == 0:
        raise _fail("Rien n'a été enregistré.")

    audio = trim_silence(to_float(audio), sr)
    report = analyze(audio, sr)
    out = get_settings().voices_dir / name / f"ref_{label}.wav"
    save_wav(out, audio, sr)
    typer.echo(f"Sauvegardé : {out} ({report.duration_s:.1f} s, {sr} Hz)")
    warnings = reference_warnings(report)
    for w in warnings:
        typer.secho(f"  ! {w}", fg=typer.colors.YELLOW)
    if not warnings:
        typer.secho("  Qualité correcte pour un clonage.", fg=typer.colors.GREEN)


@app.command("check-ref")
def check_ref(path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Analyse un fichier audio existant comme extrait de référence."""
    from voicestudio.audio import analyze, load_wav, reference_warnings

    audio, sr = load_wav(path)
    r = analyze(audio, sr)
    typer.echo(
        f"{r.duration_s:.1f} s · {r.sample_rate} Hz · crête {r.peak_dbfs:.1f} dBFS · "
        f"voix {r.speech_rms_dbfs:.1f} dBFS · silence {r.silence_ratio:.0%}"
    )
    warnings = reference_warnings(r)
    for w in warnings:
        typer.secho(f"  ! {w}", fg=typer.colors.YELLOW)
    if not warnings:
        typer.secho("  Qualité correcte pour un clonage.", fg=typer.colors.GREEN)


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False),
    lang: str | None = typer.Option(None, help="Code langue (fr, en...). Défaut : détection."),
    srt: Path | None = typer.Option(None, help="Écrit aussi des sous-titres SRT."),
    vocab: bool = typer.Option(False, help="Utiliser VOICESTUDIO_VOCAB_FILE pour guider Whisper."),
) -> None:
    """Transcrit un fichier audio ou vidéo avec Whisper (local)."""
    from voicestudio.asr import Transcriber, write_srt

    try:
        result = Transcriber().transcribe(audio, language=lang, use_vocab=vocab)
    except (RuntimeError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(result.text)
    if srt:
        typer.echo(f"SRT : {write_srt(result.segments, srt)}")


@app.command()
def translate(
    text: str = typer.Argument(...),
    source: str = typer.Option("fr", "--from"),
    target: str = typer.Option("en", "--to"),
) -> None:
    """Traduit un texte via l'API configurée dans .env."""
    from voicestudio.translate import TranslationError, Translator

    try:
        typer.echo(Translator().translate(text, source, target))
    except TranslationError as exc:
        raise _fail(str(exc)) from exc


@app.command()
def say(
    text: str = typer.Argument(...),
    backend: str = typer.Option(None, help="Moteur TTS (défaut : VOICESTUDIO_TTS_BACKEND)."),
    lang: str = typer.Option("fr"),
    voice: str | None = typer.Option(None, help="« homme », « femme » ou nom de voix."),
    reference: Path | None = typer.Option(None, help="Extrait de référence (moteurs à clonage)."),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Génère un fichier audio à partir d'un texte."""
    from voicestudio.tts import get_backend

    settings = get_settings()
    try:
        engine = get_backend(backend or settings.tts_backend)
    except (NotImplementedError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    if engine.sends_data_off_machine:
        typer.secho(
            f"Note : le moteur « {engine.name} » envoie votre texte à un service en ligne.",
            fg=typer.colors.YELLOW,
        )
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = settings.output_dir / f"say_{stamp}.wav"
    try:
        result = engine.synthesize(text, output, lang=lang, voice=voice, reference=reference)
    except (ValueError, RuntimeError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"Audio : {result}")


if __name__ == "__main__":
    app()
