"""
Speech-to-text transcription using faster-whisper.
Transcribes an audio file and returns per-segment text with timestamps.
"""

import argparse
import os
import time
from pathlib import Path

from faster_whisper import WhisperModel

# Same MODELS_DIR convention as pipeline.py, kept independent here rather
# than imported from it - pipeline.py imports FROM this module, so
# importing back would be circular.
MODELS_DIR = os.environ.get("MODELS_DIR", str(Path(__file__).parent / "models"))
DEFAULT_MODEL_PATH = str(Path(MODELS_DIR) / "faster-whisper-large-v3-turbo")

# Below this clip length, flag the result regardless of Whisper's reported
# confidence. Confidence and correctness decouple on short clips: Whisper
# can commit hard to a wrong guess in the first second or two of audio and
# report that guess back at high confidence (confirmed case: a short clip
# was misdetected at 0.88 confidence and mistranscribed as a result) since
# there just isn't enough signal yet for the model to be genuinely unsure.
# This is a separate risk signal from language_confidence, not a replacement
# for it - surface both.
SHORT_CLIP_SECONDS = 3.0


def load_asr_model(model_size: str = "large-v3-turbo", device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    """
    Load a faster-whisper model once. Reuse this instance across calls --
    model loading is the expensive part, transcription itself is cheap after that.
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(model: WhisperModel, audio_path: str, language: str = None,
                     music_mode: bool = False) -> tuple[list[dict], str, float, float]:
    """
    Transcribe an audio file and return
    (segments, detected_language_code, language_confidence, duration_seconds).

    Args:
        model: A loaded WhisperModel instance.
        audio_path: Path to the audio file to transcribe.
        language: Force a specific Whisper language code (e.g. "ne" for Nepali)
            instead of auto-detecting. Use this whenever you already know the
            spoken language - auto-detect can misfire on short clips or
            lower-resource languages (confirmed: a short Nepali clip was
            misdetected as English at only 0.32 confidence and transcribed
            as English gibberish as a result). Leave as None to auto-detect.
        music_mode: Loosens the no_speech_prob/avg_logprob noise filter
            below - singing has different acoustic properties than speech
            (sustained notes, harmonies, denser instrumentation later in a
            song) that can push Whisper's own confidence scores past the
            thresholds tuned for spoken conversation, causing later parts
            of a song to be silently dropped even after correctly reaching
            Whisper. This is a SEPARATE filter from VAD's own music_mode
            (pipeline.py's get_text_from_audio) - loosening VAD alone isn't
            enough if this filter still drops what VAD let through, so
            both need the flag for song input to make it through in full.
            Defaults to False (unchanged thresholds) so normal speech
            behavior doesn't change.

    Returns:
        A tuple of:
        - List of dicts: {"start": float, "end": float, "text": str}
        - detected_language_code: Whisper's ISO code for the spoken language
          (equals `language` if you forced one)
        - language_confidence: Whisper's confidence (0-1) in that language
          guess. Always 1.0 when `language` was forced explicitly (there was
          no guess to be confident about). Callers should surface this to
          the user rather than silently trusting a low-confidence auto-detect
          - background music, short clips, and low-resource languages are
          exactly the conditions where this number drops and the guess (and
          therefore the whole transcription) can be wrong.
        - duration_seconds: total length of the input clip. A SEPARATE risk
          signal from language_confidence - short clips can produce a
          misleadingly HIGH confidence score because there's too little
          audio for the model to register uncertainty, not because the
          guess is actually reliable. Callers should warn on this
          independently of the confidence check, not only when confidence
          is also low.
    """
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language,
        # Only matters when auto-detecting (language=None): analyzes more of
        # the audio before committing to a language guess, instead of just
        # the first segment. Bumped from 3 to 8 - short-clip/low-resource-
        # language cases (and singing, where the intro/outro can be pure
        # instrumental) need more of the file sampled before the guess is
        # trustworthy. Costs more upfront scan time; worth it for accuracy
        # in an offline tool where there's no round-trip latency pressure.
        language_detection_segments=8,
        # Prevent repetition loops (hallucinations like "oh oh oh...") during
        # instrumental sections or long silences by not feeding previous text.
        condition_on_previous_text=False,
    )

    language_confidence = info.language_probability
    duration = info.duration

    print(f"Detected language: {info.language} (confidence {language_confidence:.2f}, clip duration {duration:.2f}s)")
    if language_confidence < 0.5:
        print(f"  WARNING: low confidence - transcription may be unreliable. "
              f"If you know the spoken language, pass it explicitly instead of auto-detecting.")
    if language is None and duration < SHORT_CLIP_SECONDS:
        print(f"  WARNING: short clip ({duration:.2f}s < {SHORT_CLIP_SECONDS}s) - language "
              f"confidence can read high here even when the guess is wrong, since there's too "
              f"little audio for the model to register uncertainty. Treat the detected language "
              f"as unconfirmed regardless of the confidence score shown above.")

    # ── Noise / hallucination filter ─────────────────────────────────────
    # faster-whisper gives two per-segment quality signals:
    #   no_speech_prob : probability (0-1) that the segment has NO real speech.
    #                    > 0.6 = almost certainly noise/silence → drop it.
    #   avg_logprob    : average log-likelihood of the decoded tokens.
    #                    < -1.0 = very uncertain output → likely hallucination.
    # Both thresholds are conservative: real speech almost always scores
    # no_speech_prob < 0.4 and avg_logprob > -0.8 on clear recordings.
    # music_mode relaxes both - sung segments can legitimately land past
    # these speech-tuned cutoffs without actually being noise.
    NO_SPEECH_THRESHOLD = 0.85 if music_mode else 0.6
    AVG_LOGPROB_THRESHOLD = -1.6 if music_mode else -1.0

    results = []
    dropped = 0
    for segment in segments:
        if segment.no_speech_prob > NO_SPEECH_THRESHOLD:
            print(f"  [noise filter] dropped segment (no_speech_prob={segment.no_speech_prob:.2f}): {segment.text.strip()[:40]}")
            dropped += 1
            continue
        if segment.avg_logprob < AVG_LOGPROB_THRESHOLD:
            print(f"  [noise filter] dropped segment (avg_logprob={segment.avg_logprob:.2f}): {segment.text.strip()[:40]}")
            dropped += 1
            continue
        results.append({
            "start": segment.start,
            "end":   segment.end,
            "text":  segment.text.strip(),
        })

    if dropped:
        print(f"  Dropped {dropped} noisy segment(s), kept {len(results)}.")

    if not results:
        raise ValueError(
            "No speech detected — the recording appears to be silence or noise. "
            "Please record again with clearer audio."
        )

    return results, info.language, language_confidence, duration



def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file with faster-whisper.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Whisper model size or local folder path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--language", type=str, default=None, help="Force a Whisper language code (e.g. 'ne' for Nepali) instead of auto-detecting")
    args = parser.parse_args()

    if not Path(args.audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    compute_type = "int8" if args.device == "cpu" else "float16"

    load_start = time.time()
    model = load_asr_model(args.model, args.device, compute_type)
    load_time = time.time() - load_start

    transcribe_start = time.time()
    results, detected_language, language_confidence, duration = transcribe_audio(model, args.audio_path, args.language)
    transcribe_time = time.time() - transcribe_start

    print(f"\nTranscribed {len(results)} segment(s) [language: {detected_language}, "
          f"confidence: {language_confidence:.2f}, duration: {duration:.2f}s]:")
    for seg in results:
        print(f"  [{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")

    print(f"\nModel load time:   {load_time:.2f}s")
    print(f"Transcription time: {transcribe_time:.2f}s")


if __name__ == "__main__":
    main()