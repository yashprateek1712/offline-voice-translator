"""
Text-to-speech synthesis using Piper.
"""

import wave

from piper import PiperVoice


def load_tts_model(model_path: str, use_cuda: bool = False) -> PiperVoice:
    """
    Load a Piper voice once. Reuse across calls -- same load-once pattern
    as load_asr_model() and load_translation_model().
    """
    return PiperVoice.load(model_path, use_cuda=use_cuda)


def synthesize_speech(voice: PiperVoice, text: str, output_path: str) -> str:
    """
    Convert text to speech and save as a WAV file.
    """
    with wave.open(output_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return output_path