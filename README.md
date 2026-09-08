# Offline Translator

A fully offline, local speech-to-text, translation, and text-to-speech pipeline. This project combines Whisper for transcription, NLLB-200 for translation, and Piper TTS for voice output, all wrapped in a clean Flask UI.

## Features
- **Fully Offline**: Zero cloud dependencies. Runs entirely locally on your machine.
- **Three Input Modes**: 
  - Live microphone recording
  - Audio file upload (WAV, MP3, OGG, WebM, etc.)
  - Direct text input
- **Voice Output**: Synthesizes translated text back into speech using Piper TTS.
- **Smart Audio Processing**: Includes Voice Activity Detection (Silero-VAD) to trim silence, and a "Music Mode" optimized for translating songs.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd offline-translator
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the appropriate models for Whisper, NLLB, and Piper into a `models/` directory for the app to function.

## Usage

Start the Flask UI by running:
```bash
python ui.py
```
Then open `http://127.0.0.1:5000` in your web browser.
