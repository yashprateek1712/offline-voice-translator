# 🌐 Fully Offline Translator & Voice Synthesizer

> **Speech → Translation → Voice** — 100% offline, zero cloud, zero API keys. 
> Runs entirely locally using Whisper, NLLB-200, and Piper TTS.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web_UI-green?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

---

## 📑 Table of Contents
1. [Overview & Features](#-overview--features)
2. [How It Works (Architecture)](#-how-it-works)
3. [Model Stack & Benchmarks](#-model-stack--benchmarks)
4. [Step-by-Step Installation](#-step-by-step-installation)
5. [Usage Guide](#-usage-guide)
6. [Advanced Features](#-advanced-features)
7. [Troubleshooting](#-troubleshooting)
8. [Project Structure](#-project-structure)
9. [Limitations & Future Roadmap](#-limitations--future-roadmap)

---

## ✨ Overview & Features

This project provides a complete, end-to-end pipeline to convert spoken audio or text from any language into translated text and speech in 7 supported target languages. It is designed to be **entirely local**, ensuring maximum privacy and avoiding recurring API costs.

### Key Capabilities
- 🎙 **Live Mic Recording:** Record directly in the browser, preview your audio, and translate instantly.
- 📁 **Universal Audio Uploads:** Supports all major formats (WAV, MP3, M4A, OGG, WebM).
- ⌨️ **Direct Text Input:** For fast, text-only translation using advanced sentence-level splitting to retain grammatical structure.
- 🔊 **Neural Voice Output (TTS):** Generates natural-sounding speech in the target language.
- 🎵 **Music Mode:** Specialized toggle that disables standard Voice Activity Detection (VAD) to allow translation of singing and music without cutting out vocal tracks.
- 🛡 **Smart Hallucination Filter:** Whisper AI is prone to "hallucinating" (repeating words like "oh oh oh") during silences. This app uses strict probability thresholds and disables previous-text conditioning to guarantee clean transcripts.
- 🎨 **Glassmorphism UI:** Beautiful, modern frontend with auto-detecting Dark/Light mode and real-time audio waveforms.

---

## 🏗 How It Works

The app operates on a highly optimized pipeline. Audio is pre-processed on the CPU, transcribed and translated on the GPU (for speed), and finally synthesized back to audio on the CPU (to save VRAM).

```mermaid
graph TD
    A[User Input] --> B{Input Type}
    B -->|Live Mic / File| C[Silero VAD CPU]
    C -->|Trims Silence| D[Whisper large-v3-turbo GPU]
    D -->|Text & Lang Detect| E
    B -->|Typed Text| E[NLLB-200 1.3B GPU]
    
    E -->|Translates Sentence-by-Sentence| F[Translated Text]
    F --> G[Piper TTS CPU]
    G --> H[Synthesized Target Audio]
    
    style A fill:#2d3748,color:#fff
    style D fill:#8b5cf6,color:#fff
    style E fill:#4f9eff,color:#fff
    style G fill:#06b6d4,color:#fff
