"""
Day 1 pipeline: chain Silero-VAD -> faster-whisper.
Takes an audio file, detects speech segments, transcribes them,
and writes the combined result to a JSON file.
"""

import argparse
import json
from pathlib import Path

from vad import detect_speech_segments
from asr import load_asr_model, transcribe_audio


def run_pipeline(audio_path: str, output_path: str, model_size: str = "large-v3-turbo", device: str = "cpu") -> dict:
    print(f"Running VAD on {audio_path} ...")
    speech_segments = detect_speech_segments(audio_path)
    print(f"  Found {len(speech_segments)} speech segment(s)")

    print(f"Loading faster-whisper ({model_size}) on {device} ...")
    compute_type = "int8" if device == "cpu" else "float16"
    model = load_asr_model(model_size, device, compute_type)

    print("Transcribing ...")
    transcript_segments = transcribe_audio(model, audio_path)

    result = {
        "audio_file": audio_path,
        "vad_segments": speech_segments,
        "transcript_segments": transcript_segments,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Day 1 pipeline: VAD -> ASR")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("--output", type=str, default="output.json", help="Path to save JSON output")
    parser.add_argument("--model", type=str, default="large-v3-turbo")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if not Path(args.audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    run_pipeline(args.audio_path, args.output, args.model, args.device)


if __name__ == "__main__":
    main()
