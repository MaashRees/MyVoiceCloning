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


def _engine_and_reference(
    backend: str | None, voice: str | None, label: str | None, reference: Path | None
):
    """Choisit le moteur et trouve l'extrait de référence du profil de voix."""
    from voicestudio.tts import get_backend
    from voicestudio.voices import resolve_reference

    settings = get_settings()
    try:
        engine = get_backend(backend or settings.tts_backend, settings)
        if engine.supports_cloning and reference is None and voice:
            reference = resolve_reference(voice, label, settings)
    except (ValueError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from exc
    if engine.sends_data_off_machine:
        what = "votre texte" + (" et votre extrait de voix" if reference else "")
        typer.secho(
            f"Note : le moteur « {engine.name} » envoie {what} à un service externe.",
            fg=typer.colors.YELLOW,
        )
    if engine.supports_cloning and reference is None:
        typer.secho(
            "Aucune voix choisie (--voice ou --reference) : voix par défaut du modèle.",
            fg=typer.colors.YELLOW,
        )
    return engine, reference


def _synth_kwargs(engine, lang, voice, reference, exaggeration, cfg):
    kwargs = {"lang": lang, "voice": voice, "reference": reference}
    if engine.supports_cloning:
        kwargs.update(exaggeration=exaggeration, cfg_weight=cfg)
    return kwargs


@app.command()
def say(
    text: str = typer.Argument(...),
    backend: str = typer.Option(None, help="edge | chatterbox | remote (défaut : .env)."),
    lang: str = typer.Option("fr"),
    voice: str | None = typer.Option(None, help="Profil de voix (clonage) ou « homme »/« femme »."),
    label: str | None = typer.Option(None, help="Ambiance de l'extrait (défaut : neutre)."),
    reference: Path | None = typer.Option(None, help="Extrait de référence explicite."),
    exaggeration: float = typer.Option(0.5, help="Expressivité (clonage)."),
    cfg: float = typer.Option(0.5, help="Rythme (clonage) ; ~0.3 pour un débit rapide."),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Génère un fichier audio à partir d'un texte."""
    settings = get_settings()
    engine, reference = _engine_and_reference(backend, voice, label, reference)
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = settings.output_dir / f"say_{stamp}.wav"
    try:
        result = engine.synthesize(
            text, output, **_synth_kwargs(engine, lang, voice, reference, exaggeration, cfg)
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"Audio : {result}")


@app.command()
def voices() -> None:
    """Liste les profils de voix enregistrés."""
    from voicestudio.voices import list_voices

    found = list_voices()
    if not found:
        typer.echo("Aucun profil. Créez-en un : voicestudio record-ref moi")
    for name, labels in found.items():
        typer.echo(f"{name} : {', '.join(labels)}")


@app.command()
def convert(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="Votre prise."),
    voice: str = typer.Option(..., help="Profil de voix cible."),
    label: str | None = typer.Option(None),
    backend: str = typer.Option(None, help="chatterbox | remote (défaut : .env)."),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Voice-to-voice : garde votre intonation, prend le timbre du profil choisi."""
    settings = get_settings()
    engine, reference = _engine_and_reference(backend, voice, label, None)
    if not engine.supports_conversion or reference is None:
        raise _fail(f"Le moteur « {engine.name} » ne sait pas faire du voice-to-voice.")
    output = output or settings.output_dir / f"{audio.stem}_{voice}.wav"
    try:
        typer.echo(f"Audio : {engine.convert(audio, reference, output)}")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from exc


@app.command()
def voiceover(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    script: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Fichier texte du script."
    ),
    backend: str = typer.Option(None, help="edge | chatterbox | remote (défaut : .env)."),
    lang: str = typer.Option("fr"),
    voice: str | None = typer.Option(None, help="Profil de voix."),
    label: str | None = typer.Option(None),
    reference: Path | None = typer.Option(None),
    exaggeration: float = typer.Option(0.5),
    cfg: float = typer.Option(0.5),
    ambient: float = typer.Option(0.0, help="Volume du son d'origine gardé dessous (0 à 1)."),
    delay: float = typer.Option(0.0, help="Retard de la voix en secondes."),
    lufs: float = typer.Option(-14.0, help="Volume final visé, en LUFS."),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Génère la voix à partir du script et l'ajoute à la vidéo (ffmpeg)."""
    from voicestudio.video import add_voiceover

    settings = get_settings()
    text = script.read_text(encoding="utf-8").strip()
    engine, reference = _engine_and_reference(backend, voice, label, reference)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    voice_path = settings.output_dir / f"{video.stem}_voix_{stamp}.wav"
    output = output or video.with_name(f"{video.stem}_voiceover.mp4")
    try:
        typer.echo("Synthèse de la voix...")
        audio = engine.synthesize(
            text, voice_path, **_synth_kwargs(engine, lang, voice, reference, exaggeration, cfg)
        )
        typer.echo("Mixage avec la vidéo...")
        result = add_voiceover(video, audio, output, ambient=ambient, delay_s=delay, lufs=lufs)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"Vidéo : {result}")


@app.command()
def serve(
    port: int = typer.Option(7860),
    share: bool = typer.Option(False, help="Lien public temporaire (Colab, sans port ouvert)."),
    auth: str | None = typer.Option(None, help="« utilisateur:motdepasse » pour protéger l'accès."),
) -> None:
    """Lance le serveur de clonage (à exécuter sur la machine GPU)."""
    from voicestudio.server import build_app
    from voicestudio.tts import get_backend

    settings = get_settings()
    try:
        engine = get_backend("chatterbox", settings)
        typer.echo("Chargement du modèle (le premier lancement télécharge ~2 Go)...")
        engine.load()
        credentials = None
        if auth:
            user, sep, password = auth.partition(":")
            if not sep or not user:
                raise ValueError("--auth doit avoir la forme « utilisateur:motdepasse ».")
            credentials = (user, password)
        demo = build_app(engine)
    except (RuntimeError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=share, auth=credentials)


if __name__ == "__main__":
    app()
