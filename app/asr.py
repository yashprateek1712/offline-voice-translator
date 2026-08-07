"""
Speech-to-text transcription using faster-whisper.
Transcribes an audio file and returns per-segment text with timestamps.
"""

import argparse
from pathlib import Path

from faster_whisper import WhisperModel


def load_asr_model(model_size: str = "large-v3-turbo", device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    """
    Load a faster-whisper model once. Reuse this instance across calls --
    model loading is the expensive part, transcription itself is cheap after that.
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(model: WhisperModel, audio_path: str) -> list[dict]:
    """
    Transcribe an audio file and return a list of segment dicts.

    Args:
        model: A loaded WhisperModel instance.
        audio_path: Path to the audio file to transcribe.

    Returns:
        List of dicts: {"start": float, "end": float, "text": str}
    """
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"Detected language: {info.language} (confidence {info.language_probability:.2f})")

    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file with faster-whisper.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("--model", type=str, default="large-v3-turbo", help="Whisper model size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if not Path(args.audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    compute_type = "int8" if args.device == "cpu" else "float16"
    model = load_asr_model(args.model, args.device, compute_type)

    results = transcribe_audio(model, args.audio_path)

    print(f"\nTranscribed {len(results)} segment(s):")
    for seg in results:
        print(f"  [{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")


if __name__ == "__main__":
    main()
