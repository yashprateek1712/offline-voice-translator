"""
Full pipeline combining every stage built so far.
Accepts EITHER an audio file OR direct text as input - audio runs the full
VAD -> ASR -> Translation chain (auto-detects source language); text skips
VAD/ASR entirely and goes straight to translation (you must specify
--src-lang since there's no audio to detect it from). Always produces
translated text (printed + saved to JSON); TTS speech output is produced
whenever --voice is given.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from vad import detect_speech_segments, extract_speech_audio
from asr import load_asr_model, transcribe_audio, SHORT_CLIP_SECONDS
from translate import load_translation_model, translate_text, translate_long_text, whisper_lang_to_flores
from tts import load_tts_model, synthesize_speech

# Model root - override with the MODELS_DIR env var (e.g. a different
# drive, a Docker volume mount, wherever a deployment target keeps
# weights). Defaults to a ../models folder (one level up from this file,
# i.e. a sibling of the app/ folder this file lives in) so a fresh clone
# works out of the box as long as the models are placed there - no
# per-machine editing required.
MODELS_DIR = os.environ.get("MODELS_DIR", str(Path(__file__).parent.parent / "models"))

ASR_MODEL_PATH = os.path.join(MODELS_DIR, "faster-whisper-large-v3-turbo")
MT_MODEL_PATH = os.path.join(MODELS_DIR, "nllb-200-1.3B")

# Curated target-language menu. Source language detection (from audio) stays
# fully unrestricted via the 97-language WHISPER_TO_FLORES table - this menu
# only limits what you can translate/speak INTO, to a fixed set of 7.
#
# voice_path is None for Japanese and Korean on purpose: checked directly
# against the real rhasspy/piper-voices repository, and Piper currently has
# no voices for either language (confirmed against the actual HuggingFace
# repo listing, not assumed). Translation still works for both - only the
# spoken-audio output is unavailable until Piper adds voices for them.
LANGUAGE_MENU = {
    "1": {"name": "Hindi", "flores": "hin_Deva",
          "voice_path": os.path.join(MODELS_DIR, "piper-hi", "hi_IN-priyamvada-medium.onnx")},
    "2": {"name": "English", "flores": "eng_Latn",
          "voice_path": os.path.join(MODELS_DIR, "piper-en_US-lessac-medium", "en_US-lessac-medium.onnx")},
    "3": {"name": "Nepali", "flores": "npi_Deva",
          "voice_path": os.path.join(MODELS_DIR, "piper-ne", "ne_NP-google-medium.onnx")},
    "4": {"name": "French", "flores": "fra_Latn",
          "voice_path": os.path.join(MODELS_DIR, "piper-fr", "fr_FR-upmc-medium.onnx")},
    "5": {"name": "Russian", "flores": "rus_Cyrl",
          "voice_path": os.path.join(MODELS_DIR, "piper-ru", "ru_RU-irina-medium.onnx")},
    "6": {"name": "Japanese", "flores": "jpn_Jpan", "voice_path": None},
    "7": {"name": "Korean", "flores": "kor_Hang", "voice_path": None},
}


def prompt_target_language() -> dict:
    """
    Interactive menu - asks which of the 7 curated languages to translate
    (and speak, where a voice is available) into.
    """
    print("\nWhich language do you want to translate into?")
    for key, lang in LANGUAGE_MENU.items():
        note = "" if lang["voice_path"] else "  (text only - no Piper voice exists for this language)"
        print(f"  {key}. {lang['name']}{note}")

    choice = input("Enter a number (1-7): ").strip()
    while choice not in LANGUAGE_MENU:
        choice = input("Not a valid option - enter a number 1-7: ").strip()

    return LANGUAGE_MENU[choice]


def get_text_from_audio(audio_path: str, device: str, asr_model=None,
                        whisper_language: str = None,
                        skip_vad: bool = False, music_mode: bool = False) -> tuple[str, str, float, float]:
    """
    Run VAD + ASR on an audio file. Returns
    (full_text, detected_src_lang_flores, language_confidence, duration_seconds).

    language_confidence is Whisper's confidence (0-1) in its language guess.
    It's always 1.0 when whisper_language was forced explicitly - there was
    no guess made. Callers (api.py, ui.py) should surface this to the user
    instead of silently trusting a low-confidence auto-detect result -
    background music, short clips, and low-resource languages are exactly
    the conditions where this drops and the detected language (and
    therefore the whole transcription) can be wrong.

    duration_seconds is a SEPARATE risk signal from language_confidence, not
    a substitute for it: short clips can score misleadingly HIGH confidence
    (there's too little audio for the model to register uncertainty) while
    still being wrong. Callers should warn on short duration independently
    of the confidence check.

    skip_vad=True skips the soundfile-based VAD step and passes the audio
    directly to Whisper. Use this for browser uploads (WebM, OGG, M4A, MP3...)
    which soundfile cannot read — faster-whisper handles all these formats
    internally via its own ffmpeg integration, so VAD is the only step that
    would fail. The quality tradeoff is minimal for short web recordings.

    music_mode=True skips VAD entirely for singing/music input instead of
    just loosening it. Silero-VAD is a speech-classification model - it was
    never trained to recognize singing as speech, and no amount of
    threshold tuning reliably fixes that (confirmed directly: even with a
    loosened threshold, VAD kept only 18% of a real 212s test song,
    discarding large stretches that were genuinely sung, not instrumental).
    Rather than keep tuning a model for content it wasn't built for, music
    mode hands Whisper the full, untrimmed audio and relies solely on
    asr.py's own no_speech_prob/avg_logprob filtering (also loosened for
    music_mode) to drop genuine noise afterward. Normal speech input
    (music_mode=False) is completely unaffected - VAD trimming still runs
    exactly as before.
    """
    if asr_model is None:
        print("Loading ASR model...")
        compute_type = "int8" if device == "cpu" else "float16"
        asr_model = load_asr_model(ASR_MODEL_PATH, device, compute_type)

    audio_for_whisper = audio_path
    if music_mode:
        # See docstring above: loosening VAD's threshold wasn't enough
        # (proved directly - it still kept only 18% of a real 212s test
        # song). Skip it entirely for music mode rather than keep tuning a
        # model that isn't built to recognize singing as speech.
        print("Music mode - skipping VAD, handing Whisper the full audio.")
    elif not skip_vad:
        try:
            print("Running VAD...")
            segments = detect_speech_segments(audio_path)
            print(f"  Found {len(segments)} speech segment(s)")
            # Actually use those segments: crop out everything VAD flagged as
            # non-speech BEFORE Whisper sees it, instead of just printing the
            # count and transcribing the untouched file. Whisper hallucinates
            # fluent-sounding text on silence/noise it's handed - this is
            # what actually prevents that, not just a diagnostic step.
            audio_for_whisper = extract_speech_audio(audio_path, segments=segments)
        except Exception as e:
            print(f"  VAD failed ({e}) — falling back to direct Whisper transcription.")
            skip_vad = True   # auto-fallback for unsupported formats

    print("Transcribing...")
    results, detected_lang, language_confidence, duration = transcribe_audio(
        asr_model, audio_for_whisper, whisper_language, music_mode=music_mode
    )
    full_text = " ".join(seg["text"] for seg in results)
    last_segment_end = results[-1]["end"] if results else 0.0
    print(f"  Transcript: {len(full_text)} character(s) across {len(results)} kept segment(s), "
          f"last segment ends at {last_segment_end:.1f}s (audio handed to Whisper was {duration:.1f}s)")
    src_lang  = whisper_lang_to_flores(detected_lang)
    # language_confidence is always 1.0 here when whisper_language was forced
    # (no guess was made) - only meaningful to flag when auto-detecting.
    if whisper_language is None:
        print(f"  Detected: {detected_lang} → {src_lang} (confidence {language_confidence:.2f}, duration {duration:.2f}s)")
        if duration < SHORT_CLIP_SECONDS:
            print(f"  WARNING: short clip ({duration:.2f}s) - confidence above can read high "
                  f"even when the detected language is wrong. Treat it as unconfirmed.")
    else:
        print(f"  Forced language: {detected_lang} → {src_lang}")

    return full_text, src_lang, language_confidence, duration


def run_pipeline(text: str = None, audio_path: str = None, src_lang: str = None, tgt_lang: str = "hin_Deva",
                  voice_path: str = None, device: str = "cuda", output_prefix: str = "pipeline_output",
                  asr_model=None, mt_tokenizer=None, mt_model=None, tts_voice=None,
                  whisper_language: str = None, skip_vad: bool = False, music_mode: bool = False) -> dict:
    """
    Runs translation (+ optional TTS) starting from either text or audio input.
    skip_vad=True bypasses soundfile-based VAD — use for browser uploads (WebM,
    M4A, OGG etc.) that soundfile cannot read; Whisper handles them directly.
    music_mode=True loosens VAD for singing/music input - see
    get_text_from_audio's docstring for the reasoning and trade-off.
    """
    language_confidence = None  # only meaningful for audio input with auto-detect
    audio_duration = None       # only meaningful for audio input
    if audio_path:
        source_text, detected_src_lang, language_confidence, audio_duration = get_text_from_audio(
            audio_path, device, asr_model, whisper_language, skip_vad=skip_vad, music_mode=music_mode
        )
        src_lang = src_lang or detected_src_lang
    elif text:
        source_text = text
        if not src_lang:
            raise ValueError("--src-lang is required when using --text input (no ASR to auto-detect from)")
    else:
        raise ValueError("Provide either --text or --audio as input")

    same_language = (src_lang == tgt_lang)

    if same_language:
        # Whisper (or the manually picked source language) already matches
        # the target - there's nothing to translate. Skip calling NLLB
        # entirely rather than forcing it through a same-language
        # "translation" it isn't built for (it ends up paraphrasing
        # instead of returning the original text - see translate_text's
        # identity short-circuit). This also means the translation model
        # never has to be loaded at all for a request like this.
        print(f"Source and target are both {tgt_lang} - skipping translation.")
        translated_text = source_text
    else:
        if mt_tokenizer is None or mt_model is None:
            print(f"Loading translation model from {MT_MODEL_PATH} on {device} ...")
            mt_tokenizer, mt_model = load_translation_model(MT_MODEL_PATH, device)

        print("Translating...")
        translated_text = translate_long_text(mt_tokenizer, mt_model, source_text, src_lang, tgt_lang, device)

    result = {
        "source_text": source_text,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "translated_text": translated_text,
    }
    # Only present when audio input went through auto-detect (whisper_language
    # was not forced). api.py / ui.py use this to warn the user instead of
    # silently presenting a low-confidence language guess as fact.
    if language_confidence is not None and whisper_language is None:
        result["language_confidence"] = language_confidence
    # audio_duration_seconds / short_clip_warning are a SEPARATE risk signal
    # from language_confidence, surfaced whenever audio input was used at
    # all (even with a forced language) - short clips can score misleadingly
    # HIGH confidence while still being wrong, so this needs its own flag
    # rather than only firing alongside a low-confidence warning.
    if audio_duration is not None:
        result["audio_duration_seconds"] = audio_duration
        if whisper_language is None and audio_duration < SHORT_CLIP_SECONDS:
            result["short_clip_warning"] = (
                f"Short clip ({audio_duration:.1f}s). Language confidence can read high here "
                f"even when the detected language is wrong - there's too little audio for the "
                f"model to register uncertainty. If the result looks off, retry with a longer "
                f"recording or set src_lang explicitly."
            )

    text_output_path = f"{output_prefix}.json"
    with open(text_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved translated text to {text_output_path}")

    if voice_path:
        if same_language and audio_path:
            # The original recording already IS speech in the target
            # language - reuse it directly instead of re-synthesizing
            # through Piper. It's a real human voice (higher fidelity
            # than TTS) and skips a synthesis pass that would just
            # resay words that were never translated.
            audio_output_path = f"{output_prefix}{Path(audio_path).suffix or '.wav'}"
            shutil.copy(audio_path, audio_output_path)
            print(f"Source/target match - reused the original recording as {audio_output_path}")
        else:
            if tts_voice is None:
                print("Loading TTS voice...")
                # Piper doesn't need GPU for speed and use_cuda=True without
                # onnxruntime-gpu installed just triggers a harmless-but-noisy
                # CUDA-provider fallback warning - always run it on CPU.
                tts_voice = load_tts_model(voice_path, use_cuda=False)
            audio_output_path = f"{output_prefix}.wav"
            synthesize_speech(tts_voice, translated_text, audio_output_path)
            print(f"Saved translated speech to {audio_output_path}")
        result["audio_output"] = audio_output_path

    print(f"\nSource:     {source_text}")
    print(f"Translated: {translated_text}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: audio or text -> translated text + optional speech")
    parser.add_argument("--audio", type=str, help="Path to input audio file (uses VAD+ASR, auto-detects language)")
    parser.add_argument("--text", type=str, help="Direct text input (skips VAD+ASR)")
    parser.add_argument("--src-lang", type=str, default=None, help="Required with --text; auto-detected with --audio")
    parser.add_argument("--tgt-lang", type=str, default=None, help="FLORES-200 code. Omit to get an interactive menu of 7 curated languages instead.")
    parser.add_argument("--voice", type=str, default=None, help="Path to a Piper .onnx voice. Ignored if --tgt-lang is omitted (menu selection sets this automatically).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--output-prefix", type=str, default="../samples/pipeline_output")
    args = parser.parse_args()

    if not args.audio and not args.text:
        parser.error("Provide either --audio or --text")
    if args.audio and args.text:
        parser.error("Provide only one of --audio or --text, not both")

    if args.tgt_lang:
        tgt_lang = args.tgt_lang
        voice_path = args.voice
    else:
        selected = prompt_target_language()
        tgt_lang = selected["flores"]
        voice_path = selected["voice_path"]
        if not voice_path:
            print(f"Note: no Piper voice available for {selected['name']} - producing text output only.")

    run_pipeline(
        text=args.text,
        audio_path=args.audio,
        src_lang=args.src_lang,
        tgt_lang=tgt_lang,
        voice_path=voice_path,
        device=args.device,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()