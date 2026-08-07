"""
Voice Activity Detection using Silero-VAD.
Detects speech segments in an audio file and returns their timestamps.
"""

import argparse
from pathlib import Path

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


def detect_speech_segments(audio_path: str, sampling_rate: int = 16000) -> list[dict]:
    """
    Run Silero-VAD on an audio file and return speech segment timestamps.

    Args:
        audio_path: Path to a WAV file (mono, ideally 16kHz).
        sampling_rate: Sample rate VAD expects (Silero-VAD supports 8000 or 16000).

    Returns:
        List of dicts like {"start": 0.52, "end": 3.14} in seconds.
    """
    model = load_silero_vad()
    wav = read_audio(audio_path, sampling_rate=sampling_rate)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sampling_rate,
        return_seconds=True,
    )
    return speech_timestamps


def main():
    parser = argparse.ArgumentParser(description="Detect speech segments in an audio file.")
    parser.add_argument("audio_path", type=str, help="Path to input WAV file")
    args = parser.parse_args()

    if not Path(args.audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    segments = detect_speech_segments(args.audio_path)

    if not segments:
        print("No speech detected.")
        return

    print(f"Detected {len(segments)} speech segment(s):")
    for i, seg in enumerate(segments, start=1):
        print(f"  [{i}] {seg['start']:.2f}s -> {seg['end']:.2f}s")


if __name__ == "__main__":
    main()
