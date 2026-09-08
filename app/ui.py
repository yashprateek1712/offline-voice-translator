"""
Reslate-inspired Flask UI for the Offline Translator.
Run: python ui.py  →  open http://127.0.0.1:5000
"""

import importlib
import os
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string

import vad
import asr
import translate
import tts
import pipeline
from pipeline import LANGUAGE_MENU, ASR_MODEL_PATH, MT_MODEL_PATH, run_pipeline
from asr import load_asr_model
from translate import load_translation_model
from tts import load_tts_model

DEVICE     = "cuda"
OUTPUT_DIR = Path(os.environ.get("SAMPLES_DIR", str(Path(__file__).parent / "samples")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("⏳ Loading ASR model...")
_asr = load_asr_model(ASR_MODEL_PATH, DEVICE, "float16" if DEVICE == "cuda" else "int8")
print("⏳ Loading translation model...")
_tok, _mt = load_translation_model(MT_MODEL_PATH, DEVICE)
print("⏳ Loading voice models...")
_voices = {}
for k, v in LANGUAGE_MENU.items():
    if v.get("voice_path"):
        _voices[k] = load_tts_model(v["voice_path"], use_cuda=False)
print("✅ All models ready!\n")

app = Flask(__name__)


@app.route("/reload-code")
def reload_code():
    """
    DEV ONLY. Hot-reloads the CODE in vad/asr/translate/tts/pipeline after
    you edit one of those files, WITHOUT reloading the models those files
    loaded once at startup — _asr, _tok, _mt, and every voice in _voices
    stay exactly as they are in memory. That's the slow, multi-GB part;
    this route only swaps out functions/logic, which is instant.

    After saving an edit, hit http://127.0.0.1:5000/reload-code in your
    browser (or `curl http://127.0.0.1:5000/reload-code`) instead of
    Ctrl+C-ing and rerunning `python ui.py` and waiting for every model
    to load all over again.

    Reload order matters: leaf modules first (vad, asr, translate, tts),
    then pipeline last, so pipeline's own `from vad import ...`-style
    imports pick up the freshly reloaded versions instead of stale ones.

    Caveat: this does NOT help if you changed how a model is LOADED
    (load_asr_model, load_translation_model, load_tts_model) or added a
    new LANGUAGE_MENU entry that needs its own voice loaded — those need
    an actual restart, since this route deliberately never re-runs any
    loading code.
    """
    global run_pipeline, LANGUAGE_MENU, ASR_MODEL_PATH, MT_MODEL_PATH
    importlib.reload(vad)
    importlib.reload(asr)
    importlib.reload(translate)
    importlib.reload(tts)
    importlib.reload(pipeline)
    run_pipeline   = pipeline.run_pipeline
    LANGUAGE_MENU  = pipeline.LANGUAGE_MENU
    ASR_MODEL_PATH = pipeline.ASR_MODEL_PATH
    MT_MODEL_PATH  = pipeline.MT_MODEL_PATH
    return jsonify({"status": "reloaded", "note": "code refreshed — models were NOT touched"})

SPOKEN_LANGS = {
    "auto": "Auto-detect",
    "en": "English", "hi": "Hindi", "ne": "Nepali",
    "fr": "French",  "ru": "Russian", "ja": "Japanese",
    "ko": "Korean",  "es": "Spanish", "de": "German",
    "ar": "Arabic",  "zh": "Chinese",
}


HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Offline Translator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* ── Tokens ── */
:root {
  --c-bg:       #070b14;
  --c-surface:  rgba(255,255,255,0.04);
  --c-surface2: rgba(255,255,255,0.07);
  --c-border:   rgba(255,255,255,0.08);
  --c-border-hi:rgba(139,92,246,0.5);
  --c-text:     #e2e8f5;
  --c-muted:    #6b7a99;
  --c-violet:   #8b5cf6;
  --c-blue:     #4f9eff;
  --c-cyan:     #06b6d4;
  --c-red:      #ef4444;
  --c-green:    #22c55e;
  --grad:       linear-gradient(135deg, #8b5cf6, #4f9eff, #06b6d4);
  --grad-btn:   linear-gradient(135deg, #7c3aed, #2563eb);
  --blur:       blur(20px);
}
[data-theme="light"] {
  --c-bg:       #f0f4ff;
  --c-surface:  rgba(255,255,255,0.7);
  --c-surface2: rgba(255,255,255,0.9);
  --c-border:   rgba(139,92,246,0.15);
  --c-border-hi:rgba(139,92,246,0.4);
  --c-text:     #0f172a;
  --c-muted:    #64748b;
}

/* ── Reset ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body {
  font-family:'Inter',sans-serif;
  background:var(--c-bg);
  color:var(--c-text);
  min-height:100vh;
  overflow-x:hidden;
  transition:background .4s,color .4s;
}

/* ── Background glow orbs ── */
body::before,body::after {
  content:'';position:fixed;border-radius:50%;filter:blur(120px);
  pointer-events:none;z-index:0;
}
body::before {
  width:600px;height:600px;
  background:radial-gradient(circle,rgba(139,92,246,.18),transparent 70%);
  top:-200px;left:-200px;
}
body::after {
  width:500px;height:500px;
  background:radial-gradient(circle,rgba(6,182,212,.12),transparent 70%);
  bottom:-150px;right:-150px;
}
[data-theme="light"] body::before { background:radial-gradient(circle,rgba(139,92,246,.08),transparent 70%); }
[data-theme="light"] body::after  { background:radial-gradient(circle,rgba(6,182,212,.06),transparent 70%); }

/* ── Nav ── */
nav {
  position:sticky;top:0;z-index:200;
  display:flex;align-items:center;justify-content:space-between;
  padding:18px 40px;
  background:rgba(7,11,20,.6);
  backdrop-filter:var(--blur);
  border-bottom:1px solid var(--c-border);
}
[data-theme="light"] nav { background:rgba(240,244,255,.7); }
.brand {
  display:flex;align-items:center;gap:10px;
  font-size:1rem;font-weight:700;letter-spacing:-.02em;
}
.brand-icon {
  width:32px;height:32px;border-radius:8px;
  background:var(--grad-btn);
  display:flex;align-items:center;justify-content:center;
  font-size:.9rem;
}
.nav-r { display:flex;align-items:center;gap:12px; }
.badge-local {
  font-size:.7rem;font-weight:600;padding:4px 12px;border-radius:999px;
  background:rgba(34,197,94,.12);color:var(--c-green);
  border:1px solid rgba(34,197,94,.25);letter-spacing:.03em;
}
/* Theme toggle */
.toggle-wrap {
  display:flex;align-items:center;gap:8px;
  background:var(--c-surface2);border:1px solid var(--c-border);
  border-radius:999px;padding:5px 10px;cursor:pointer;
  font-size:.82rem;color:var(--c-muted);transition:all .2s;
}
.toggle-wrap:hover{border-color:var(--c-border-hi)}
.toggle-pill {
  width:36px;height:20px;border-radius:999px;
  background:var(--grad-btn);position:relative;transition:background .3s;
}
.toggle-pill::after {
  content:'';position:absolute;top:3px;left:3px;
  width:14px;height:14px;border-radius:50%;background:#fff;
  transition:transform .3s;
}
[data-theme="light"] .toggle-pill::after { transform:translateX(16px); }

/* ── Hero ── */
.hero {
  position:relative;z-index:1;
  text-align:center;padding:64px 20px 40px;
}
.hero h1 {
  font-size:clamp(2.4rem,6vw,4rem);
  font-weight:800;letter-spacing:-.04em;line-height:1.1;
  background:var(--grad);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
  margin-bottom:14px;
}
.hero p { color:var(--c-muted);font-size:1rem;margin-bottom:20px; }
.hero-chips { display:flex;gap:8px;justify-content:center;flex-wrap:wrap; }
.chip {
  font-size:.72rem;font-weight:500;padding:5px 14px;border-radius:999px;
  background:var(--c-surface2);border:1px solid var(--c-border);
  color:var(--c-muted);backdrop-filter:var(--blur);
}

/* ── Main cards ── */
.cards-wrap {
  position:relative;z-index:1;
  display:grid;grid-template-columns:1fr 1fr;
  gap:20px;max-width:960px;margin:0 auto;padding:0 24px 60px;
}
@media(max-width:700px){.cards-wrap{grid-template-columns:1fr}}

.glass-card {
  background:var(--c-surface);
  border:1px solid var(--c-border);
  border-radius:24px;
  padding:28px;
  backdrop-filter:var(--blur);
  box-shadow:0 0 0 1px rgba(139,92,246,.05), 0 24px 48px rgba(0,0,0,.35);
  transition:border-color .3s,box-shadow .3s;
}
.glass-card:hover {
  border-color:var(--c-border-hi);
  box-shadow:0 0 0 1px rgba(139,92,246,.15), 0 0 40px rgba(139,92,246,.08), 0 24px 48px rgba(0,0,0,.35);
}
[data-theme="light"] .glass-card {
  box-shadow:0 4px 24px rgba(139,92,246,.08),0 1px 3px rgba(0,0,0,.06);
}

/* ── Tabs ── */
.tabs {
  display:inline-flex;gap:4px;
  background:var(--c-surface2);border:1px solid var(--c-border);
  border-radius:14px;padding:5px;margin-bottom:24px;width:100%;
}
.tab {
  flex:1;padding:9px 0;border:none;border-radius:10px;
  cursor:pointer;font-size:.83rem;font-weight:500;
  background:transparent;color:var(--c-muted);
  transition:all .2s;font-family:'Inter',sans-serif;
}
.tab.active {
  background:var(--grad-btn);color:#fff;
  box-shadow:0 2px 12px rgba(124,58,237,.4);
}
.tab:hover:not(.active){color:var(--c-text)}

.pane{display:none}.pane.active{display:block}

/* ── Mic button ── */
.mic-center {
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  padding:20px 0 28px;
}
.mic-btn-wrap { position:relative;margin-bottom:16px; }
.mic-ring {
  position:absolute;inset:-16px;border-radius:50%;
  border:2px solid rgba(239,68,68,.3);
  animation:none;
}
.mic-ring.recording { animation:ring-pulse 1.2s ease-in-out infinite; }
.mic-ring2 {
  position:absolute;inset:-30px;border-radius:50%;
  border:2px solid rgba(239,68,68,.15);
  animation:none;
}
.mic-ring2.recording { animation:ring-pulse 1.2s ease-in-out infinite .4s; }
@keyframes ring-pulse {
  0%,100%{transform:scale(1);opacity:1}
  50%{transform:scale(1.08);opacity:.4}
}
.mic-btn {
  width:90px;height:90px;border-radius:50%;border:none;
  background:radial-gradient(circle at 35% 35%, #1e1e2e, #0d0d1a);
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;position:relative;z-index:1;
  box-shadow:0 0 0 3px rgba(239,68,68,.25), 0 8px 32px rgba(0,0,0,.5);
  transition:all .2s;
}
.mic-btn:hover { transform:scale(1.06); box-shadow:0 0 0 3px rgba(239,68,68,.45),0 8px 32px rgba(0,0,0,.5); }
.mic-btn.recording {
  background:radial-gradient(circle at 35% 35%, #3b0a0a, #1a0505);
  box-shadow:0 0 0 3px rgba(239,68,68,.6), 0 0 30px rgba(239,68,68,.3), 0 8px 32px rgba(0,0,0,.5);
}
.mic-btn.has-audio {
  background:radial-gradient(circle at 35% 35%, #0a2a1a, #051510);
  box-shadow:0 0 0 3px rgba(34,197,94,.4), 0 0 20px rgba(34,197,94,.15), 0 8px 32px rgba(0,0,0,.5);
}
.mic-svg { width:36px;height:36px;color:#ef4444;transition:color .2s; }
.mic-btn.has-audio .mic-svg { color:#22c55e; }
.mic-status { font-size:.85rem;color:var(--c-muted);font-weight:500; }
.mic-timer  { font-size:1.4rem;font-weight:700;color:#ef4444;margin-top:4px;display:none; }
.mic-actions { display:flex;gap:8px;margin-top:16px; }

/* ── Upload zone ── */
.upload-zone {
  border:2px dashed var(--c-border);border-radius:16px;padding:32px;
  text-align:center;cursor:pointer;transition:all .25s;margin-bottom:16px;
}
.upload-zone:hover,.upload-zone.drag {
  border-color:var(--c-violet);
  background:rgba(139,92,246,.05);
}
.upload-zone.has-file { border-color:var(--c-green); background:rgba(34,197,94,.04); }
.upload-icon { font-size:2rem;margin-bottom:8px; }
.upload-hint { font-size:.83rem;color:var(--c-muted); }
#file-input { display:none; }

/* ── Form ── */
label { display:block;font-size:.78rem;font-weight:500;color:var(--c-muted);margin-bottom:6px;letter-spacing:.02em; }
.field { margin-bottom:16px; }
textarea,.sel {
  width:100%;background:var(--c-surface2);border:1px solid var(--c-border);
  border-radius:12px;padding:12px 14px;color:var(--c-text);
  font-size:.94rem;font-family:'Inter',sans-serif;
  transition:border-color .2s,box-shadow .2s;resize:vertical;
  backdrop-filter:var(--blur);
}
textarea:focus,.sel:focus {
  outline:none;border-color:var(--c-violet);
  box-shadow:0 0 0 3px rgba(139,92,246,.15);
}
.sel { cursor:pointer; }

/* ── Language selector row ── */
.lang-row {
  display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px;
}
@media(max-width:480px){.lang-row{grid-template-columns:1fr}}

/* ── Divider ── */
.divider { display:flex;align-items:center;gap:10px;margin:18px 0; }
.divider::before,.divider::after { content:'';flex:1;height:1px;background:var(--c-border); }
.divider span { color:var(--c-muted);font-size:.72rem;white-space:nowrap;letter-spacing:.06em;text-transform:uppercase; }

/* ── Translate button ── */
.btn-translate {
  width:100%;padding:14px;border:none;border-radius:14px;
  background:var(--grad-btn);color:#fff;
  font-size:1rem;font-weight:700;font-family:'Inter',sans-serif;
  cursor:pointer;letter-spacing:-.01em;
  box-shadow:0 4px 24px rgba(124,58,237,.35);
  transition:all .2s;position:relative;overflow:hidden;
}
.btn-translate::before {
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.1),transparent);
  opacity:0;transition:opacity .2s;
}
.btn-translate:hover::before { opacity:1; }
.btn-translate:hover { transform:translateY(-1px);box-shadow:0 6px 30px rgba(124,58,237,.45); }
.btn-translate:active { transform:translateY(0); }
.btn-translate:disabled { opacity:.4;cursor:not-allowed;transform:none; }

/* ── Small buttons ── */
.btn-sm {
  background:var(--c-surface2);border:1px solid var(--c-border);
  border-radius:10px;padding:8px 16px;cursor:pointer;
  font-size:.82rem;font-weight:500;color:var(--c-text);
  font-family:'Inter',sans-serif;transition:all .2s;
}
.btn-sm:hover { border-color:var(--c-border-hi);background:rgba(139,92,246,.08); }

/* ── Output card ── */
.card-label {
  font-size:.72rem;font-weight:600;color:var(--c-muted);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:16px;
}
.det-badge {
  display:inline-flex;align-items:center;gap:6px;
  font-size:.72rem;font-weight:500;padding:4px 12px;border-radius:999px;
  background:rgba(79,158,255,.1);color:var(--c-blue);
  border:1px solid rgba(79,158,255,.2);margin-bottom:14px;
  opacity:0;transition:opacity .3s;
}
.det-badge.show { opacity:1; }
.det-badge.low-confidence {
  background:rgba(239,68,68,.1);color:var(--c-red);
  border-color:rgba(239,68,68,.25);
}

.output-text {
  min-height:120px;font-size:1.1rem;line-height:1.7;
  color:var(--c-text);padding:4px 0;white-space:pre-wrap;
  transition:all .3s;
}
.output-text.empty { font-size:.94rem;color:var(--c-muted);font-style:italic; }

.out-actions { display:flex;gap:8px;margin-top:14px;justify-content:flex-end; }

/* ── Waveform audio player ── */
.audio-section { display:none;margin-top:20px; }
.audio-section.visible { display:block; }
.audio-player-wrap {
  background:var(--c-surface2);border:1px solid var(--c-border);
  border-radius:16px;padding:16px 18px;
  display:flex;align-items:center;gap:14px;
}
.play-btn {
  width:42px;height:42px;border-radius:50%;border:none;
  background:var(--grad-btn);color:#fff;
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;flex-shrink:0;font-size:1rem;
  box-shadow:0 4px 16px rgba(124,58,237,.4);transition:all .2s;
}
.play-btn:hover { transform:scale(1.08); }
.waveform {
  flex:1;height:36px;border-radius:8px;overflow:hidden;position:relative;
  background:rgba(79,158,255,.06);
}
.waveform canvas { width:100%;height:100%; }
.audio-time { font-size:.78rem;color:var(--c-muted);white-space:nowrap;font-variant-numeric:tabular-nums; }
audio { display:none; }

/* ── Spinner ── */
.spinner-wrap { display:none;flex-direction:column;align-items:center;gap:14px;padding:40px 0; }
.spinner-wrap.active { display:flex; }
.spinner {
  width:40px;height:40px;border-radius:50%;
  border:3px solid var(--c-border);
  border-top-color:var(--c-violet);
  animation:spin .75s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg); } }
.spin-txt { color:var(--c-muted);font-size:.84rem; }

/* ── Toast ── */
#toast {
  position:fixed;bottom:28px;right:28px;
  padding:13px 22px;border-radius:12px;font-size:.86rem;font-weight:500;
  opacity:0;pointer-events:none;transition:all .3s;z-index:999;
  backdrop-filter:var(--blur);
}
#toast.show { opacity:1;transform:translateY(0); }
#toast.success { background:rgba(34,197,94,.9);color:#fff; }
#toast.error   { background:rgba(239,68,68,.9);color:#fff; }
</style>
</head>
<body>

<!-- Nav -->
<nav>
  <div class="brand">
    <div class="brand-icon">🌐</div>
    Offline Translator
  </div>
  <div class="nav-r">
    <span class="badge-local">⚡ GPU · Local</span>
    <div class="toggle-wrap" id="theme-toggle">
      <span id="theme-icon">🌙</span>
      <div class="toggle-pill" id="toggle-pill"></div>
      <span id="theme-label">Dark</span>
    </div>
  </div>
</nav>

<!-- Hero -->
<div class="hero">
  <h1>Speak. Translate. Listen.</h1>
  <p>Whisper + NLLB-200 + Piper TTS &mdash; fully offline, zero cloud</p>
  <div class="hero-chips">
    <span class="chip">🎙 Live mic</span>
    <span class="chip">📁 Audio file</span>
    <span class="chip">⌨️ Text input</span>
    <span class="chip">🔊 Voice output</span>
    <span class="chip">🌍 7 languages</span>
    <span class="chip">🔒 Private</span>
  </div>
</div>

<!-- Cards -->
<div class="cards-wrap">

  <!-- INPUT CARD -->
  <div class="glass-card">
    <div class="card-label">Input</div>

    <div class="tabs">
      <button class="tab active" onclick="switchTab('mic',this)">🎙 Mic</button>
      <button class="tab" onclick="switchTab('file',this)">📁 File</button>
      <button class="tab" onclick="switchTab('text',this)">⌨️ Text</button>
    </div>

    <!-- Mic pane -->
    <div class="pane active" id="pane-mic">
      <div class="mic-center">
        <div class="mic-btn-wrap">
          <div class="mic-ring"  id="mic-ring"></div>
          <div class="mic-ring2" id="mic-ring2"></div>
          <button class="mic-btn" id="mic-btn" onclick="toggleRecord()">
            <svg class="mic-svg" id="mic-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="23"/>
              <line x1="8"  y1="23" x2="16" y2="23"/>
            </svg>
          </button>
        </div>
        <div class="mic-status" id="mic-status">Click to start recording</div>
        <div class="mic-timer"  id="mic-timer">00:00</div>
        <div class="mic-actions">
          <button class="btn-sm" id="mic-action-btn" onclick="toggleRecord()">⏺ Start</button>
          <button class="btn-sm" id="mic-clear-btn" onclick="clearMic()" style="display:none;color:#ef4444;border-color:rgba(239,68,68,.3)">🗑 Clear</button>
        </div>
        <audio id="mic-preview" controls style="display:none;margin-top:14px;width:100%;max-width:340px"></audio>
      </div>

      <div class="field">
        <label>Spoken language</label>
        <select class="sel" id="spoken-mic">
          {% for code,name in spoken_langs.items() %}
          <option value="{{ code }}">{{ name }}</option>
          {% endfor %}
        </select>
      </div>
      <label style="display:flex;align-items:center;gap:8px;font-size:.82rem;font-weight:400;color:var(--c-muted);cursor:pointer;margin-bottom:16px">
        <input type="checkbox" id="music-mode-mic" style="width:15px;height:15px;cursor:pointer">
        🎵 This is a song/music (loosens speech detection to catch singing)
      </label>
    </div>

    <!-- File pane -->
    <div class="pane" id="pane-file">
      <div class="upload-zone" id="upload-zone"
           onclick="document.getElementById('file-input').click()"
           ondragover="event.preventDefault();this.classList.add('drag')"
           ondragleave="this.classList.remove('drag')"
           ondrop="handleDrop(event)">
        <div class="upload-icon">📂</div>
        <div class="upload-hint" id="upload-lbl">Click or drag &amp; drop audio file<br><small style="font-size:.72rem;opacity:.6">WAV, MP3, OGG, WebM...</small></div>
        <input type="file" id="file-input" accept="audio/*" onchange="handleFile(this.files[0])">
      </div>
      <audio id="file-preview" controls style="display:none;margin-top:14px;width:100%"></audio>
      <div class="field">
        <label>Spoken language</label>
        <select class="sel" id="spoken-file">
          {% for code,name in spoken_langs.items() %}
          <option value="{{ code }}">{{ name }}</option>
          {% endfor %}
        </select>
      </div>
      <label style="display:flex;align-items:center;gap:8px;font-size:.82rem;font-weight:400;color:var(--c-muted);cursor:pointer;margin-bottom:16px">
        <input type="checkbox" id="music-mode-file" style="width:15px;height:15px;cursor:pointer">
        🎵 This is a song/music (loosens speech detection to catch singing)
      </label>
    </div>

    <!-- Text pane -->
    <div class="pane" id="pane-text">
      <div class="field">
        <label>Your text <span style="font-weight:400;opacity:.6">&nbsp;Ctrl+Enter to translate</span></label>
        <textarea id="text-input" rows="6" placeholder="Type or paste text here..."></textarea>
      </div>
      <div class="field">
        <label>Text language</label>
        <select class="sel" id="src-lang">
          {% for key,lang in languages.items() %}
          <option value="{{ lang.flores }}">{{ lang.name }}</option>
          {% endfor %}
        </select>
      </div>
    </div>

    <div class="divider"><span>translate into</span></div>

    <div class="field">
      <label>Target language</label>
      <select class="sel" id="tgt-lang">
        {% for key,lang in languages.items() %}
        <option value="{{ key }}">{{ lang.name }}{% if not lang.has_voice %} · text only{% endif %}</option>
        {% endfor %}
      </select>
    </div>

    <button class="btn-translate" id="translate-btn" onclick="doTranslate()">
      ⚡&nbsp; Translate
    </button>
  </div>

  <!-- OUTPUT CARD -->
  <div class="glass-card">
    <div class="card-label">Output</div>

    <div class="det-badge" id="det-badge">🔍 <span id="det-txt">—</span></div>

    <div class="spinner-wrap" id="spinner">
      <div class="spinner"></div>
      <div class="spin-txt" id="spin-txt">Translating...</div>
    </div>

    <div class="output-text empty" id="out-text">Your translation will appear here...</div>

    <div class="out-actions" id="out-actions" style="display:none">
      <button class="btn-sm" onclick="copyText()">📋 Copy</button>
      <button class="btn-sm" onclick="dlText()">⬇ Save</button>
    </div>

    <!-- Custom audio player -->
    <div class="audio-section" id="audio-section">
      <div class="divider"><span>audio output</span></div>
      <div class="audio-player-wrap">
        <button class="play-btn" id="play-btn" onclick="togglePlay()">▶</button>
        <div class="waveform">
          <canvas id="waveform-canvas"></canvas>
        </div>
        <span class="audio-time" id="audio-time">0:00</span>
        <button class="btn-sm" onclick="dlAudio()" style="padding:8px 12px">⬇</button>
      </div>
      <audio id="audio-el"></audio>
    </div>
  </div>

</div><!-- /cards-wrap -->

<div id="toast"></div>

<script>
/* ── Theme ── */
const html = document.documentElement;
function setTheme(t) {
  html.dataset.theme = t;
  document.getElementById('theme-icon').textContent  = t==='dark'?'🌙':'☀️';
  document.getElementById('theme-label').textContent = t==='dark'?'Dark':'Light';
  localStorage.setItem('theme', t);
}
setTheme(localStorage.getItem('theme') || (matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light'));
document.getElementById('theme-toggle').onclick = () => setTheme(html.dataset.theme==='dark'?'light':'dark');

/* ── Tabs ── */
let currentTab = 'mic';
function switchTab(n, btn) {
  currentTab = n;
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('pane-'+n).classList.add('active');
}

/* ── Mic ── */
let recorder, chunks=[], timerInt, secs=0, recBlob=null;
const pad = n => String(n).padStart(2,'0');
function toggleRecord() { (!recorder||recorder.state==='inactive') ? startRec() : stopRec(); }
async function startRec() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    recorder = new MediaRecorder(stream);
    chunks=[]; secs=0;
    // Starting a fresh recording - drop any leftover preview from a
    // previous take so it doesn't linger while the new one is captured.
    const preview = document.getElementById('mic-preview');
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.removeAttribute('src');
    preview.style.display = 'none';
    recorder.ondataavailable = e=>chunks.push(e.data);
    recorder.onstop = () => {
      recBlob = new Blob(chunks, {type: recorder.mimeType||'audio/webm'});
      // Let you actually hear back exactly what was captured before
      // sending it off to translate - same as Gradio's audio preview.
      preview.src = URL.createObjectURL(recBlob);
      preview.style.display = 'block';
      setMicState('done');
    };
    recorder.start(100);
    setMicState('recording');
    timerInt = setInterval(()=>{
      secs++;
      document.getElementById('mic-timer').textContent=`${pad(Math.floor(secs/60))}:${pad(secs%60)}`;
    }, 1000);
  } catch(e) { toast('Mic access denied','error'); }
}
function stopRec() {
  if(recorder){ recorder.stop(); recorder.stream.getTracks().forEach(t=>t.stop()); }
  clearInterval(timerInt);
}
function setMicState(state) {
  const btn=document.getElementById('mic-btn');
  const ring=document.getElementById('mic-ring');
  const ring2=document.getElementById('mic-ring2');
  const status=document.getElementById('mic-status');
  const timer=document.getElementById('mic-timer');
  const ab=document.getElementById('mic-action-btn');
  const cb=document.getElementById('mic-clear-btn');

  btn.className='mic-btn'+(state==='recording'?' recording':state==='done'?' has-audio':'');
  ring.className='mic-ring'+(state==='recording'?' recording':'');
  ring2.className='mic-ring2'+(state==='recording'?' recording':'');

  if(state==='recording'){
    status.textContent='Recording… click Stop when done';
    timer.style.display='block';
    ab.textContent='⏹ Stop';
    cb.style.display='none';
  } else if(state==='done'){
    status.textContent=`Recorded ${pad(Math.floor(secs/60))}:${pad(secs%60)} ✓`;
    timer.style.display='none';
    ab.textContent='⏺ Record again';
    cb.style.display='';
  } else {
    status.textContent='Click to start recording';
    timer.style.display='none';
    timer.textContent='00:00';
    ab.textContent='⏺ Start';
    cb.style.display='none';
  }
}
function clearMic() {
  recBlob=null; secs=0; recorder=null; clearInterval(timerInt);
  const preview = document.getElementById('mic-preview');
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.removeAttribute('src');
  preview.style.display = 'none';
  setMicState('idle');
}

/* ── File ── */
let upFile=null;
function handleFile(f) {
  if(!f) return;
  upFile=f;
  document.getElementById('upload-lbl').innerHTML=`✅ <strong>${f.name}</strong><br><small style="opacity:.6">${(f.size/1024).toFixed(0)} KB</small>`;
  document.getElementById('upload-zone').classList.add('has-file');
  // Play back the exact file that's about to be sent - same as Gradio's
  // audio preview, so a wrong file pick is obvious before you translate.
  const preview = document.getElementById('file-preview');
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(f);
  preview.style.display = 'block';
}
function handleDrop(ev) { ev.preventDefault(); document.getElementById('upload-zone').classList.remove('drag'); const f=ev.dataTransfer.files[0]; if(f)handleFile(f); }

/* ── Translate ── */
let lastText='', lastAudioUrl=null;
async function doTranslate() {
  const tgt=document.getElementById('tgt-lang').value;
  const btn=document.getElementById('translate-btn');
  const spin=document.getElementById('spinner');
  const spinTxt=document.getElementById('spin-txt');
  const fd=new FormData();
  fd.append('tgt_lang_key', tgt);

  if(currentTab==='mic'){
    if(!recBlob){toast('Record audio first!','error');return;}
    const ext=recBlob.type.includes('ogg')?'ogg':'webm';
    fd.append('file', recBlob, `rec.${ext}`);
    fd.append('spoken', document.getElementById('spoken-mic').value);
    fd.append('music_mode', document.getElementById('music-mode-mic').checked);
    spinTxt.textContent='Transcribing + Translating…';
  } else if(currentTab==='file'){
    if(!upFile){toast('Select a file first!','error');return;}
    fd.append('file', upFile);
    fd.append('spoken', document.getElementById('spoken-file').value);
    fd.append('music_mode', document.getElementById('music-mode-file').checked);
    spinTxt.textContent='Transcribing + Translating…';
  } else {
    const txt=document.getElementById('text-input').value.trim();
    if(!txt){toast('Enter some text!','error');return;}
    fd.append('text', txt);
    fd.append('src_lang', document.getElementById('src-lang').value);
    spinTxt.textContent='Translating…';
  }

  btn.disabled=true; spin.classList.add('active');
  document.getElementById('out-text').textContent='';
  document.getElementById('out-text').className='output-text empty';
  document.getElementById('out-actions').style.display='none';
  document.getElementById('audio-section').classList.remove('visible');
  document.getElementById('det-badge').classList.remove('show');

  try {
    const res=await fetch('/translate',{method:'POST',body:fd});
    const data=await res.json();
    if(data.error){toast(data.error,'error');return;}
    lastText=data.translated_text||'';
    const el=document.getElementById('out-text');
    el.textContent=lastText; el.className='output-text';
    document.getElementById('out-actions').style.display='flex';
    if(data.src_lang){
      const badge = document.getElementById('det-badge');
      const txt = document.getElementById('det-txt');
      if(data.language_confidence !== undefined){
        const pct = Math.round(data.language_confidence*100);
        txt.textContent = `Detected: ${data.src_lang} (${pct}% confidence)`;
        badge.classList.toggle('low-confidence', !!data.language_low_confidence);
        if(data.language_low_confidence){
          toast(`Low confidence (${pct}%) detecting the spoken language — pick it manually if this looks wrong`, 'error');
        }
      } else {
        txt.textContent = 'Detected: '+data.src_lang;
        badge.classList.remove('low-confidence');
      }
      // Separate from the confidence check above: a short clip can score
      // HIGH confidence and still be wrong, so this fires independently
      // rather than only alongside the low-confidence toast.
      if(data.short_clip_warning){
        badge.classList.add('low-confidence');
        toast(data.short_clip_warning, 'error');
      }
      badge.classList.add('show');
    }
    if(data.audio_url){
      lastAudioUrl=data.audio_url;
      const ael=document.getElementById('audio-el');
      ael.src=data.audio_url+'?t='+Date.now(); ael.load();
      document.getElementById('audio-section').classList.add('visible');
      drawWaveform();
      // No autoplay - shouldn't blast audio the moment a translation
      // finishes. togglePlay() (bound to #play-btn) is the only way
      // playback starts now.
      document.getElementById('play-btn').textContent='▶';
    }
    toast('Done! ✅','success');
  } catch(e){ toast('Server error','error'); console.error(e); }
  finally { btn.disabled=false; spin.classList.remove('active'); }
}

/* ── Custom audio player ── */
function togglePlay() {
  const a=document.getElementById('audio-el');
  const btn=document.getElementById('play-btn');
  if(a.paused){ a.play(); btn.textContent='⏸'; }
  else { a.pause(); btn.textContent='▶'; }
}
document.getElementById('audio-el').addEventListener('ended',()=>{ document.getElementById('play-btn').textContent='▶'; });
document.getElementById('audio-el').addEventListener('timeupdate', ()=>{
  const a=document.getElementById('audio-el');
  const t=a.currentTime; const m=Math.floor(t/60); const s=Math.floor(t%60);
  document.getElementById('audio-time').textContent=`${m}:${pad(s)}`;
});

function drawWaveform() {
  const canvas=document.getElementById('waveform-canvas');
  const ctx=canvas.getContext('2d');
  canvas.width=canvas.offsetWidth||300; canvas.height=36;
  const bars=40; const gap=2; const w=(canvas.width-(bars-1)*gap)/bars;
  const grad=ctx.createLinearGradient(0,0,canvas.width,0);
  grad.addColorStop(0,'#8b5cf6'); grad.addColorStop(.5,'#4f9eff'); grad.addColorStop(1,'#06b6d4');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let i=0;i<bars;i++){
    const h=8+Math.random()*20;
    const x=i*(w+gap); const y=(36-h)/2;
    ctx.fillStyle=grad;
    ctx.beginPath();
    ctx.roundRect(x,y,w,h,2);
    ctx.fill();
  }
}

function copyText(){ navigator.clipboard.writeText(lastText).then(()=>toast('Copied!','success')); }
function dlText(){ const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([lastText],{type:'text/plain'})); a.download='translation.txt'; a.click(); }
function dlAudio(){ if(!lastAudioUrl)return; const a=document.createElement('a'); a.href=lastAudioUrl; a.download='translation.wav'; a.click(); }

let toastT;
function toast(msg,type='success'){
  const el=document.getElementById('toast');
  el.textContent=msg; el.className=`show ${type}`;
  clearTimeout(toastT); toastT=setTimeout(()=>el.className='',3200);
}

document.getElementById('text-input').addEventListener('keydown', e=>{ if(e.ctrlKey&&e.key==='Enter') doTranslate(); });
</script>
</body>
</html>"""


@app.route("/")
def index():
    langs = {k: {"name": v["name"], "flores": v["flores"],
                 "has_voice": bool(v.get("voice_path"))}
             for k, v in LANGUAGE_MENU.items()}
    return render_template_string(HTML, languages=langs, spoken_langs=SPOKEN_LANGS)


@app.route("/translate", methods=["POST"])
def translate():
    tgt_key = request.form.get("tgt_lang_key")
    if tgt_key not in LANGUAGE_MENU:
        return jsonify({"error": f"Invalid language key: {tgt_key}"})

    selected  = LANGUAGE_MENU[tgt_key]
    tgt_lang  = selected["flores"]
    tts_voice = _voices.get(tgt_key)
    spoken    = request.form.get("spoken", "auto")
    whisper_lang = None if spoken == "auto" else spoken
    music_mode = request.form.get("music_mode", "false").lower() == "true"

    audio_path = None

    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        suffix = Path(f.filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
            audio_path = tmp.name
        # Do NOT force skip_vad=True here: get_text_from_audio() already
        # falls back to raw audio on its own when soundfile can't read a
        # format (webm, m4a, ogg...) - forcing it here unconditionally
        # used to also disable VAD trimming for formats that DO work fine
        # (e.g. an uploaded .wav), silently giving up the hallucination
        # fix for those cases too.
    elif not request.form.get("text"):
        return jsonify({"error": "Provide audio file or text"})

    try:
        result = run_pipeline(
            text=request.form.get("text") if not audio_path else None,
            audio_path=audio_path,
            src_lang=request.form.get("src_lang"),
            tgt_lang=tgt_lang,
            voice_path=selected.get("voice_path"),
            device=DEVICE,
            output_prefix=str(OUTPUT_DIR / "ui_out"),
            asr_model=_asr, mt_tokenizer=_tok, mt_model=_mt,
            tts_voice=tts_voice,
            whisper_language=whisper_lang,
            music_mode=music_mode,
        )
    except ValueError as e:
        return jsonify({"error": str(e)})
    finally:
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)

    resp = {
        "source_text":     result.get("source_text", ""),
        "src_lang":        result.get("src_lang", ""),
        "translated_text": result.get("translated_text", ""),
    }
    if "audio_output" in result:
        resp["audio_url"] = f"/audio/{Path(result['audio_output']).name}"

    # Only present for auto-detected audio input (not text, not a forced
    # spoken-language selection). Surfaced to the browser instead of only
    # printed server-side, so a low-confidence guess is visible to the user
    # rather than presented as a certain result.
    LOW_CONFIDENCE_THRESHOLD = 0.5
    if "language_confidence" in result:
        resp["language_confidence"] = result["language_confidence"]
        resp["language_low_confidence"] = result["language_confidence"] < LOW_CONFIDENCE_THRESHOLD

    # A SEPARATE signal from language_confidence above: short clips can
    # score misleadingly HIGH confidence (too little audio for the model to
    # register uncertainty) while still landing on the wrong language, so
    # the frontend needs this even when language_low_confidence is False.
    if "audio_duration_seconds" in result:
        resp["audio_duration_seconds"] = result["audio_duration_seconds"]
    if "short_clip_warning" in result:
        resp["short_clip_warning"] = result["short_clip_warning"]

    return jsonify(resp)



_AUDIO_MIMETYPES = {
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
}


@app.route("/audio/<filename>")
def serve_audio(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    # audio_output is usually a Piper-synthesized .wav, but when source and
    # target language matched, pipeline.py now reuses the ORIGINAL
    # recording verbatim (typically the browser's .webm) instead of
    # resynthesizing it - serve it with its real mimetype rather than
    # hardcoding audio/wav, which would mislabel it and can make some
    # browsers refuse playback.
    mimetype = _AUDIO_MIMETYPES.get(path.suffix.lower(), "audio/wav")
    return send_file(str(path), mimetype=mimetype, conditional=True)


if __name__ == "__main__":
    print("🌐 Open http://127.0.0.1:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)