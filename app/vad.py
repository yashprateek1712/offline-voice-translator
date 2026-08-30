"""
Voice Activity Detection using Silero-VAD.
Detects speech segments in an audio file and returns their timestamps.

Reads audio with soundfile directly instead of Silero-VAD's read_audio()
helper, which goes through torchaudio's file-loading backend - unreliable
on Windows (missing sox, a torchcodec requirement, and torchcodec's own
FFmpeg DLL dependencies). Reading the file ourselves avoids all of that.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from silero_vad import load_silero_vad, get_speech_timestamps

TARGET_SAMPLE_RATE = 16000  # Silero-VAD only supports 8kHz or 16kHz

# Container formats libsndfile (soundfile's backend) cannot open at all.
# Browser mic recordings (MediaRecorder) come out as .webm/.ogg with an
# Opus codec, and phone/voice-memo uploads are often .m4a/.mp3 - none of
# these are in libsndfile's supported set, so sf.read() raises
# "Format not recognised" for every one of them.
_LIBSNDFILE_UNSUPPORTED = {".webm", ".m4a", ".mp3", ".mp4", ".aac"}


def _find_ffmpeg() -> str:
    """
    Locate an ffmpeg binary. Prefers the bundled imageio-ffmpeg binary over
    a system install - a system/conda ffmpeg can be found on PATH but still
    fail to launch if one of its own DLL dependencies doesn't resolve,
    which shows up as ffmpeg exiting instantly with no output. The bundled
    binary has no external DLLs to resolve, so it can't fail that way.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    raise RuntimeError(
        "No ffmpeg found. Run `pip install imageio-ffmpeg` for a bundled "
        "binary that needs no separate install."
    )


def _transcode_to_wav(audio_path: str, target_sample_rate: int) -> str:
    """
    Transcode an arbitrary audio file to mono 16-bit PCM WAV at
    target_sample_rate using ffmpeg, and return the path to the result.
    """
    ffmpeg_exe = _find_ffmpeg()
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    # mkstemp() also hands back an open file descriptor. Close it
    # immediately - on Windows, leaving it open makes soundfile report the
    # file as "in use by another process" once ffmpeg writes to it below.
    os.close(fd)
    try:
        subprocess.run(
            [
                ffmpeg_exe, "-y", "-i", audio_path,
                "-ac", "1",                       # mono - VAD only needs one channel
                "-ar", str(target_sample_rate),   # resample to what Silero-VAD expects
                wav_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to transcode {audio_path}: {e.stderr.decode(errors='ignore')}")
    return wav_path


def load_audio_as_tensor(audio_path: str, target_sample_rate: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """
    Load an audio file as a mono float32 tensor at target_sample_rate.

    Tries soundfile first; falls back to an ffmpeg transcode for formats
    libsndfile can't read (webm, m4a, mp3 - notably browser mic
    recordings), so VAD can actually run on those instead of always being
    skipped.
    """
    suffix = Path(audio_path).suffix.lower()
    tmp_wav = None
    read_path = audio_path

    if suffix in _LIBSNDFILE_UNSUPPORTED:
        tmp_wav = _transcode_to_wav(audio_path, target_sample_rate)
        read_path = tmp_wav

    try:
        try:
            audio, sample_rate = sf.read(read_path, dtype="float32", always_2d=True)
        except Exception:
            if tmp_wav is not None:
                raise  # already transcoded and still failed - don't retry
            # Extension wasn't in our known-bad list but soundfile still
            # choked on it - fall back to ffmpeg instead of giving up.
            tmp_wav = _transcode_to_wav(audio_path, target_sample_rate)
            audio, sample_rate = sf.read(tmp_wav, dtype="float32", always_2d=True)

        wav = torch.from_numpy(audio.T)  # (channels, samples)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # downmix to mono

        if sample_rate != target_sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, target_sample_rate)

        return wav.squeeze(0)  # (samples,)
    finally:
        if tmp_wav is not None:
            Path(tmp_wav).unlink(missing_ok=True)


def detect_speech_segments(audio_path: str, sampling_rate: int = TARGET_SAMPLE_RATE) -> list[dict]:
    """
    Run Silero-VAD on an audio file and return speech segment timestamps.

    Args:
        audio_path: Path to a WAV file (mono, ideally 16kHz).
        sampling_rate: Sample rate VAD expects (Silero-VAD supports 8000 or 16000).

    Returns:
        List of dicts like {"start": 0.52, "end": 3.14} in seconds.
    """
    model = load_silero_vad()
    wav = load_audio_as_tensor(audio_path, sampling_rate)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sampling_rate,
        return_seconds=True,
    )
    return speech_timestamps


def extract_speech_audio(audio_path: str, segments: list[dict] | None = None,
                         sampling_rate: int = TARGET_SAMPLE_RATE, padding: float = 0.15):
    """
    Return only the detected speech portions of the audio, concatenated
    into one array - avoids feeding Whisper long stretches of silence or
    noise, which it can transcribe as fluent-sounding hallucinated text.
    `padding` keeps a small buffer around each segment so words at a
    boundary aren't clipped. Falls back to the full audio if nothing was
    detected, rather than returning an empty array.
    """
    wav = load_audio_as_tensor(audio_path, sampling_rate)

    if segments is None:
        model = load_silero_vad()
        segments = get_speech_timestamps(wav, model, sampling_rate=sampling_rate, return_seconds=True)

    if not segments:
        return wav.numpy()

    pad_samples = int(padding * sampling_rate)
    total_samples = wav.shape[0]
    chunks = []
    for seg in segments:
        start = max(0, int(seg["start"] * sampling_rate) - pad_samples)
        end = min(total_samples, int(seg["end"] * sampling_rate) + pad_samples)
        chunks.append(wav[start:end])

    return torch.cat(chunks).numpy()


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