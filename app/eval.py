"""
Full evaluation suite for the offline translator.
Computes:
  Translation -> BLEU, chrF, TER  (sacrebleu)
  ASR         -> WER, CER         (jiwer)

Install deps first:
    pip install sacrebleu jiwer

Run:
    conda activate translator
    python eval.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import sacrebleu
except ImportError:
    print("Run: pip install sacrebleu"); sys.exit(1)
try:
    import jiwer
except ImportError:
    print("Run: pip install jiwer"); sys.exit(1)

from pipeline import MT_MODEL_PATH, ASR_MODEL_PATH
from translate import load_translation_model, translate_text
from asr import load_asr_model, transcribe_audio


# ── Translation test pairs (English source → Hindi reference) ─────────────────
MT_TEST_PAIRS = [
    ("Hello, how are you?",
     "नमस्ते, आप कैसे हैं?"),
    ("The weather is very nice today.",
     "आज मौसम बहुत अच्छा है।"),
    ("I am learning machine translation.",
     "मैं मशीन अनुवाद सीख रहा हूँ।"),
    ("This project runs completely offline on a GPU.",
     "यह परियोजना GPU पर पूरी तरह से ऑफलाइन चलती है।"),
    ("Natural language processing is a fascinating field.",
     "प्राकृतिक भाषा प्रसंस्करण एक आकर्षक क्षेत्र है।"),
    ("Please speak clearly into the microphone.",
     "कृपया माइक्रोफोन में स्पष्ट रूप से बोलें।"),
    ("The translation was completed successfully.",
     "अनुवाद सफलतापूर्वक पूरा हो गया।"),
    ("I want to translate this sentence into Hindi.",
     "मैं इस वाक्य का हिंदी में अनुवाद करना चाहता हूँ।"),
    ("Good morning, have a great day.",
     "सुप्रभात, आपका दिन शुभ हो।"),
    ("The model is running on a GPU for faster inference.",
     "मॉडल तेज़ अनुमान के लिए GPU पर चल रहा है।"),
    ("She went to the market to buy vegetables.",
     "वह सब्ज़ियाँ खरीदने बाज़ार गई।"),
    ("The student passed the exam with good marks.",
     "छात्र ने अच्छे अंकों से परीक्षा पास की।"),
    ("Water is essential for all living beings.",
     "पानी सभी जीवित प्राणियों के लिए आवश्यक है।"),
    ("He is reading a book in the library.",
     "वह पुस्तकालय में एक किताब पढ़ रहा है।"),
    ("The train arrived at the station on time.",
     "ट्रेन समय पर स्टेशन पर पहुँची।"),
]

# ASR test pairs - add your own WAV files here
ASR_TEST_PAIRS = [
    # ("samples/test1.wav", "hello how are you"),
    # ("samples/test2.wav", "my name is prateek"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

def bleu_grade(s):
    if s >= 40: return "Excellent"
    if s >= 25: return "Good"
    if s >= 15: return "Moderate"
    return "Low (normal for 1.3B model)"

def chrf_grade(s):
    if s >= 55: return "Excellent"
    if s >= 40: return "Good"
    if s >= 25: return "Moderate"
    return "Low"

def wer_grade(s):
    if s <= 0.05: return "Excellent (<5%)"
    if s <= 0.10: return "Good (<10%)"
    if s <= 0.20: return "Moderate (<20%)"
    return "High (>20%)"


# ── 1. Translation Evaluation ─────────────────────────────────────────────────
banner("TRANSLATION  NLLB-200-1.3B  English to Hindi")
print("\nLoading translation model...")
tok, mt = load_translation_model(MT_MODEL_PATH, "cuda")
print("Ready\n")

hypotheses, references = [], []
for i, (src, ref) in enumerate(MT_TEST_PAIRS, 1):
    hyp = translate_text(tok, mt, src, "eng_Latn", "hin_Deva", "cuda")
    hypotheses.append(hyp)
    references.append(ref)
    print(f"[{i:02d}] SRC: {src}")
    print(f"      REF: {ref}")
    print(f"      HYP: {hyp}\n")

bleu = sacrebleu.corpus_bleu(hypotheses, [references])
chrf = sacrebleu.corpus_chrf(hypotheses, [references])
ter  = sacrebleu.corpus_ter(hypotheses, [references])

print(f"\n  BLEU : {bleu.score:.2f}   {bleu_grade(bleu.score)}")
print(f"  chrF : {chrf.score:.2f}   {chrf_grade(chrf.score)}  <- best for Hindi")
print(f"  TER  : {ter.score:.2f}   Lower is better")


# ── 2. ASR Evaluation ─────────────────────────────────────────────────────────
if ASR_TEST_PAIRS:
    banner("ASR  Whisper large-v3-turbo")
    print("\nLoading Whisper model...")
    asr = load_asr_model(ASR_MODEL_PATH, "cuda", "float16")
    print("Ready\n")

    all_refs, all_hyps = [], []
    for wav_path, reference in ASR_TEST_PAIRS:
        if not os.path.exists(wav_path):
            print(f"  Skipping missing: {wav_path}"); continue
        segments, _, _, _ = transcribe_audio(asr, wav_path, language="en")
        hyp = " ".join(s["text"].strip() for s in segments)
        all_refs.append(reference.lower())
        all_hyps.append(hyp.lower())
        print(f"  REF: {reference}")
        print(f"  HYP: {hyp}")
        print(f"  WER: {jiwer.wer(reference.lower(), hyp.lower())*100:.1f}%\n")

    if all_refs:
        tf = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase(),
                            jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
        wer = jiwer.wer(all_refs, all_hyps, reference_transform=tf, hypothesis_transform=tf)
        cer = jiwer.cer(all_refs, all_hyps)
        print(f"\n  WER : {wer*100:.1f}%   {wer_grade(wer)}")
        print(f"  CER : {cer*100:.1f}%   Character-level")
else:
    banner("ASR EVALUATION SKIPPED")
    print("\n  To run ASR eval, add WAV files to ASR_TEST_PAIRS at the top.")


# ── Final README table ────────────────────────────────────────────────────────
banner("COPY THIS INTO YOUR README")
print(f"""
## Evaluation Results

### Translation (NLLB-200-1.3B, English to Hindi, {len(MT_TEST_PAIRS)} sentences)

| Metric | Score | Interpretation |
|--------|------:|----------------|
| BLEU   | {bleu.score:.1f}  | {bleu_grade(bleu.score)} |
| chrF   | {chrf.score:.1f}  | {chrf_grade(chrf.score)} |
| TER    | {ter.score:.1f}   | Edit distance, lower = better |

chrF compares character n-grams instead of whole words, so it's less
harsh on the compound words and minor spelling variation common in
Devanagari - generally a better fit for Hindi than word-level BLEU.
""")