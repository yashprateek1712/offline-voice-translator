"""
Benchmark script - measures latency of each pipeline component.
Run once, copy numbers to the README.

Usage:
    conda activate translator
    python benchmark.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import LANGUAGE_MENU, ASR_MODEL_PATH, MT_MODEL_PATH
from asr import load_asr_model, transcribe_audio
from translate import load_translation_model, translate_text
from tts import load_tts_model, synthesize_speech

DEVICE = "cuda"
RUNS   = 3  # average over N runs for stability

if DEVICE == "cuda" and not torch.cuda.is_available():
    print("CUDA requested but not available on this machine. "
          "Set DEVICE = \"cpu\" at the top of this file, or install a "
          "CUDA-enabled PyTorch build.")
    sys.exit(1)

GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
GPU_VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if torch.cuda.is_available() else None
)

TEST_SENTENCES = [
    "Hello, how are you today?",
    "The weather is very nice in the morning.",
    "I am learning machine translation systems.",
    "This project runs completely offline on a GPU.",
    "Natural language processing is a fascinating field.",
]

SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"


def timeit(fn, runs=RUNS):
    """Run fn() N times, return (last_result, avg_ms, min_ms, max_ms)."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return result, sum(times) / len(times), min(times), max(times)


def section(title):
    print(f"\n{'-'*55}")
    print(f"  {title}")
    print(f"{'-'*55}")


# 1. ASR model load
section("1/4  Whisper model load")
t0 = time.perf_counter()
asr = load_asr_model(ASR_MODEL_PATH, DEVICE, "float16")
load_asr_ms = (time.perf_counter() - t0) * 1000
print(f"  Load time : {load_asr_ms/1000:.2f}s")

# 2. ASR inference - benchmark against real synthesized speech, not a tone.
# A pure sine wave isn't speech, and asr.py's own noise filter is built to
# reject exactly that kind of input (no_speech_prob / avg_logprob
# thresholds), so timing Whisper on a tone risks tripping that guard
# entirely, and even when it doesn't, timing a hallucinated transcription
# doesn't measure real transcription latency. Synthesizing a short English
# sentence with Piper first gives Whisper genuine speech to transcribe,
# using models already loaded elsewhere in this script.
section("2/4  Whisper transcription (synthesized speech clip)")
english_voice_path = LANGUAGE_MENU["2"]["voice_path"]
voice_for_test = load_tts_model(english_voice_path, use_cuda=False)
test_wav_path = "benchmark_test.wav"
BENCHMARK_SENTENCE = "The quick brown fox jumps over the lazy dog near the river."
synthesize_speech(voice_for_test, BENCHMARK_SENTENCE, test_wav_path)

_, asr_avg, asr_min, asr_max = timeit(
    lambda: transcribe_audio(asr, test_wav_path, language="en"), runs=RUNS
)
os.remove(test_wav_path)
print(f"  Test sentence : \"{BENCHMARK_SENTENCE}\"")
print(f"  Avg : {asr_avg:.0f} ms")
print(f"  Min : {asr_min:.0f} ms")
print(f"  Max : {asr_max:.0f} ms")

# 3. MT model load
section("3/4  NLLB-200 1.3B translation")
t0 = time.perf_counter()
tok, mt = load_translation_model(MT_MODEL_PATH, DEVICE)
load_mt_ms = (time.perf_counter() - t0) * 1000
print(f"  Load time : {load_mt_ms/1000:.2f}s")

times_mt = []
for sent in TEST_SENTENCES:
    _, avg, mn, mx = timeit(
        lambda s=sent: translate_text(tok, mt, s, SRC_LANG, TGT_LANG, DEVICE),
        runs=RUNS
    )
    times_mt.append(avg)
    word_count = len(sent.split())
    print(f"  {word_count:2d} words -> {avg:.0f}ms avg  [{mn:.0f}-{mx:.0f}ms]  \"{sent[:45]}\"")

overall_avg = sum(times_mt) / len(times_mt)
print(f"\n  Overall avg per sentence : {overall_avg:.0f} ms")

# 4. TTS
section("4/4  Piper TTS (CPU)")
hi_voice = LANGUAGE_MENU["1"]["voice_path"]
tts = load_tts_model(hi_voice, use_cuda=False)
tts_out = "benchmark_tts_out.wav"

test_tts_text = "नमस्ते, आप कैसे हैं? मुझे आशा है कि आप ठीक हैं।"
_, tts_avg, tts_min, tts_max = timeit(
    lambda: synthesize_speech(tts, test_tts_text, tts_out), runs=RUNS
)
if os.path.exists(tts_out):
    os.remove(tts_out)
print(f"  Input : \"{test_tts_text[:50]}\"")
print(f"  Avg   : {tts_avg:.0f} ms")
print(f"  Min   : {tts_min:.0f} ms")
print(f"  Max   : {tts_max:.0f} ms")

# Summary
section("SUMMARY  (copy this to your README)")
total_pipeline = asr_avg + overall_avg + tts_avg
hardware_line = (
    f"{GPU_NAME} ({GPU_VRAM_GB:.0f}GB VRAM)" if GPU_VRAM_GB else GPU_NAME
)
print(f"""
  Hardware  : {hardware_line}
  Device    : {DEVICE.upper()} (float16 for Whisper + NLLB)

  Component            Latency (avg)
  --------------------------------------
  Whisper load        {load_asr_ms/1000:.1f}s
  Whisper (transcribe) {asr_avg:.0f} ms
  NLLB load            {load_mt_ms/1000:.1f}s
  NLLB per sentence    {overall_avg:.0f} ms
  Piper TTS (CPU)      {tts_avg:.0f} ms
  --------------------------------------
  Full pipeline        ~{total_pipeline:.0f} ms

  Runs averaged over {RUNS} iterations.
""")