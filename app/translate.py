"""
Text-to-text translation using NLLB-200.
Translates transcript segments (Day 1's output) from one language to another.

NLLB uses FLORES-200 language codes (e.g. "eng_Latn", "hin_Deva"),
NOT ISO 639-1 codes like "en"/"hi". Full list:
https://github.com/facebookresearch/flores/blob/main/flores200/README.md
"""

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Maps faster-whisper's detected language code to the FLORES-200 code NLLB
# expects. Built from faster-whisper's real _LANGUAGE_CODES list cross-checked
# against NLLB's real FAIRSEQ_LANGUAGE_CODES - covers 97 of Whisper's 100
# language codes (Breton, Hawaiian, and Latin have no FLORES-200 equivalent).
# A few entries default to the most common script/dialect variant
# (e.g. "zh" -> Simplified Chinese, "ar" -> Modern Standard Arabic,
# "no" -> Norwegian Bokmal) - override with --src-lang if you need a
# different variant.
WHISPER_TO_FLORES = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "as": "asm_Beng",
    "az": "azj_Latn", "ba": "bak_Cyrl", "be": "bel_Cyrl", "bg": "bul_Cyrl",
    "bn": "ben_Beng", "bo": "bod_Tibt", "bs": "bos_Latn", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn",
    "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn",
    "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fo": "fao_Latn",
    "fr": "fra_Latn", "gl": "glg_Latn", "gu": "guj_Gujr", "ha": "hau_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "ht": "hat_Latn",
    "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn", "is": "isl_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "jw": "jav_Latn", "ka": "kat_Geor",
    "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang",
    "lb": "ltz_Latn", "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn",
    "lv": "lvs_Latn", "mg": "plt_Latn", "mi": "mri_Latn", "mk": "mkd_Cyrl",
    "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva", "ms": "zsm_Latn",
    "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn",
    "nn": "nno_Latn", "no": "nob_Latn", "oc": "oci_Latn", "pa": "pan_Guru",
    "pl": "pol_Latn", "ps": "pbt_Arab", "pt": "por_Latn", "ro": "ron_Latn",
    "ru": "rus_Cyrl", "sa": "san_Deva", "sd": "snd_Arab", "si": "sin_Sinh",
    "sk": "slk_Latn", "sl": "slv_Latn", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "su": "sun_Latn", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "tg": "tgk_Cyrl",
    "th": "tha_Thai", "tk": "tuk_Latn", "tl": "tgl_Latn", "tr": "tur_Latn",
    "tt": "tat_Cyrl", "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn",
    "vi": "vie_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn", "yue": "yue_Hant",
    "zh": "zho_Hans",
}


def whisper_lang_to_flores(whisper_code: str) -> str:
    """
    Convert a faster-whisper detected language code (e.g. "hi") into the
    FLORES-200 code NLLB needs (e.g. "hin_Deva"). Raises a clear error if
    the language isn't in the table, rather than silently guessing.
    """
    if whisper_code not in WHISPER_TO_FLORES:
        raise ValueError(
            f"No FLORES-200 mapping for detected language '{whisper_code}'. "
            f"Pass --src-lang manually with the correct FLORES-200 code."
        )
    return WHISPER_TO_FLORES[whisper_code]


def load_translation_model(model_name: str = "facebook/nllb-200-1.3B", device: str = "cpu"):
    """
    Load the NLLB tokenizer and model once. Reuse across calls --
    same load-once-reuse pattern as load_asr_model() from Day 1.

    Default here matches the local model folder this project actually uses
    (nllb-200-1.3B) - pass model_name explicitly (e.g. a local folder path)
    to override, same as every other load_*_model() function in this project.

    Loads in float16 on GPU (roughly halves VRAM vs the float32 default) -
    matters for fitting the larger nllb-200-1.3B on a 6GB card. CPU stays
    float32 since most CPUs don't have dedicated float16 acceleration.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(device)
    return tokenizer, model


# Unicode block each FLORES-200 script suffix should mostly fall within.
# Used to catch the case where text is typed in the WRONG script for its
# declared language - most commonly romanized/"Hinglish"-style text
# ("abhi na jao") tagged as hin_Deva. NLLB has no training data for
# romanized Hindi (there's no such FLORES-200 code), so instead of
# failing it just generates a fluent, completely made-up translation -
# this is what turns into "poor translation quality" reports that have
# nothing to do with model quality at all.
_SCRIPT_UNICODE_RANGES = {
    "Latn": (0x0041, 0x024F),   # Latin + Latin Extended - English, French...
    "Deva": (0x0900, 0x097F),   # Devanagari - Hindi, Nepali...
    "Cyrl": (0x0400, 0x04FF),   # Cyrillic - Russian
    "Hang": (0xAC00, 0xD7A3),   # Hangul - Korean
    "Arab": (0x0600, 0x06FF),
    "Grek": (0x0370, 0x03FF),
    "Hebr": (0x0590, 0x05FF),
}
_CJK_RANGE = (0x4E00, 0x9FFF)  # Han/Kanji, shared with Japanese's Jpan script


def detect_script_mismatch(text: str, flores_lang: str, threshold: float = 0.3) -> str | None:
    """
    Returns a warning message if `text` doesn't look like it's actually
    written in the script `flores_lang` expects (e.g. hin_Deva -> should be
    Devanagari), or None if it looks fine. Too-short input (<4 letters) is
    never flagged - not enough signal to judge reliably.
    """
    script = flores_lang.split("_")[-1]
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 4:
        return None

    if script == "Jpan":
        in_script = sum(
            1 for c in alpha_chars
            if 0x3040 <= ord(c) <= 0x30FF or _CJK_RANGE[0] <= ord(c) <= _CJK_RANGE[1]
        )
    elif script in _SCRIPT_UNICODE_RANGES:
        lo, hi = _SCRIPT_UNICODE_RANGES[script]
        in_script = sum(1 for c in alpha_chars if lo <= ord(c) <= hi)
    else:
        return None  # no rule for this script - don't guess, let it through

    if (in_script / len(alpha_chars)) < threshold:
        return (
            f"This text doesn't look like it's written in the script '{flores_lang}' needs. "
            f"NLLB requires native script, not romanized text - typing Hindi in Latin letters "
            f"(e.g. \"aap kaise ho\" instead of \"आप कैसे हो\") has no equivalent in its training "
            f"data, so instead of erroring out it invents a fluent-sounding but unrelated "
            f"translation. Retype the source text in its native script, or pick a target that "
            f"actually matches what you typed."
        )
    return None


def detect_language_mismatch(text: str, flores_lang: str, min_confidence: float = 0.8) -> str | None:
    """
    Catches what detect_script_mismatch structurally can't: text in the
    RIGHT script but the WRONG language. Devanagari is shared by Hindi and
    Nepali; Latin is shared by English, French, and others in this
    project's menu - a unicode-range check can't tell those apart, since
    it only looks at which script a character belongs to, not which
    language it's actually spelling out.

    Uses langdetect for the guess (this is optional - if langdetect isn't installed,
    the check is silently skipped rather than blocking translation over a
    missing dependency; `pip install langdetect` in the same env to enable
    it).

    Only flags a mismatch when langdetect is quite confident (default
    0.8) AND the text is long enough to trust that confidence. Short text
    is exactly where general-purpose language ID falls over on closely
    related languages sharing a script - verified directly: langdetect
    reads a 13-letter Russian greeting ("Привет, как дела?") as Macedonian
    at 99.99% confidence, but gets genuinely Russian sentences right once
    they're past ~25 letters. Confidence alone doesn't catch this (it was
    just as "confident" and wrong), so length is the real gate here - this
    mirrors asr.py's own caution around Whisper's language guess
    (info.language_probability < 0.5 there), just tuned for a different
    failure mode.
    """
    try:
        from langdetect import detect_langs
    except ImportError:
        return None

    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 20:
        return None  # too short for language ID between related languages to be trustworthy

    try:
        best = detect_langs(text)[0]
    except Exception:
        return None  # langdetect can raise on some inputs - don't block translation over it

    if best.prob < min_confidence:
        return None

    guessed_flores = WHISPER_TO_FLORES.get(best.lang)
    if guessed_flores and guessed_flores != flores_lang:
        return (
            f"This text looks like it might actually be '{guessed_flores}' "
            f"(langdetect confidence {best.prob:.0%}), not '{flores_lang}' as selected. "
            f"Some languages share a script - Devanagari is used by both Hindi and "
            f"Nepali, Latin by English and French - so the same letters can belong to "
            f"more than one language. Double check the Text language dropdown, or "
            f"proceed anyway if this guess is wrong for your text."
        )
    return None


def translate_text(tokenizer, model, text: str, src_lang: str, tgt_lang: str, device: str = "cpu", max_length: int = 256) -> str:
    """
    Translate a single string from src_lang to tgt_lang (FLORES-200 codes).

    If src_lang == tgt_lang, there's nothing to translate - text is
    returned unchanged without ever touching the model. NLLB isn't built
    as an identity/copy model, and no_repeat_ngram_size + repetition_penalty
    (needed to stop real cross-language translation from degenerating into
    loops) actively fight against it reproducing text that's supposed to
    look like its own input, so same-language "translation" was coming
    back as an unwanted paraphrase instead of the original text.

    no_repeat_ngram_size and repetition_penalty guard against degenerate
    repetition - a known failure mode where the decoder gets stuck picking
    the same token/phrase over and over, especially on long or unusually
    repetitive input (song lyrics, chants, anything with repeated sections).

    Raises ValueError if `text` doesn't look like it's actually written in
    the script `src_lang` claims (detect_script_mismatch), or if it looks
    confidently like a DIFFERENT language that happens to share that
    script (detect_language_mismatch) - both are surfaced through the
    same ValueError handling api.py and ui.py already use, so they show
    up as a clear message instead of a hallucinated translation.
    """
    if src_lang == tgt_lang:
        return text

    mismatch = detect_script_mismatch(text, src_lang)
    if mismatch:
        raise ValueError(mismatch)

    lang_mismatch = detect_language_mismatch(text, src_lang)
    if lang_mismatch:
        raise ValueError(lang_mismatch)

    tokenizer.src_lang = src_lang

    inputs = tokenizer(text, return_tensors="pt").to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=max_length,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
    )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation


def translate_long_text(tokenizer, model, text: str, src_lang: str, tgt_lang: str, device: str = "cpu", max_length: int = 256) -> str:
    """
    Translate text of any length safely - this is what pipeline.py (and
    therefore api.py and ui.py, which both route every request through
    run_pipeline()) has always imported and called.

    translate_text() passes max_length=256 straight to model.generate() -
    that's a hard cap on the OUTPUT the decoder is allowed to produce.
    Anything longer (a full spoken paragraph from ASR, a pasted block of
    text) gets cut off mid-sentence, which reads exactly like "the
    translation is bad" even though the model did fine on what it was
    given. This splits long input into safely-sized chunks first,
    translates each chunk with translate_text(), and joins the results
    back together.

    Splitting happens in two passes:
      1. Split on sentence-ending punctuation across the scripts this
         project's 7-language menu covers (., !, ? for Latin/Cyrillic;
         । for the Devanagari full stop used by Hindi and Nepali).
      2. Any resulting "sentence" that's STILL over the limit (very common
         in lyrics/informal speech with few or no periods, like song
         transcripts) gets hard-split further: first on commas, and if
         that still isn't enough, on raw word boundaries. This guarantees
         no chunk handed to translate_text() ever exceeds CHUNK_CHAR_LIMIT,
         so the 256-token output cap is never asked to compress more text
         than it can actually hold - which is what was producing garbled,
         truncated translations on longer inputs.
    """
    # NLLB's SentencePiece tokenizer runs roughly 1 token per 3-4
    # characters for the languages in this project's menu. Keeping chunks
    # under ~400 characters keeps each one comfortably clear of the
    # 256-token cap on both the input and the generated output.
    CHUNK_CHAR_LIMIT = 400

    text = text.strip()

    if src_lang == tgt_lang:
        return text

    if len(text) <= CHUNK_CHAR_LIMIT:
        return translate_text(tokenizer, model, text, src_lang, tgt_lang, device, max_length)

    # Pass 1: split on sentence-ending punctuation.
    sentences = re.split(r'(?<=[.!?।])\s+', text)

    def hard_split(s: str, limit: int) -> list[str]:
        """
        Break an over-long 'sentence' (common in lyrics/informal speech
        with no periods) down further - first on commas, then on raw word
        boundaries - so nothing over `limit` chars ever reaches
        translate_text().
        """
        if len(s) <= limit:
            return [s]

        parts = re.split(r'(?<=,)\s+', s)
        if len(parts) > 1 and all(len(p) <= limit for p in parts):
            return parts

        words, out, cur = s.split(), [], ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if cur and len(cand) > limit:
                out.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            out.append(cur)
        return out

    # Pass 2: hard-split anything still too long after Pass 1.
    normalized = []
    for sentence in sentences:
        normalized.extend(hard_split(sentence, CHUNK_CHAR_LIMIT))

    # Pass 3: greedily re-merge the (now safely-sized) pieces back up into
    # ~400-char chunks, so nearby sentences still translate together with
    # shared context instead of one-at-a-time.
    chunks, current = [], ""
    for sentence in normalized:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > CHUNK_CHAR_LIMIT:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)

    print(f"  translate_long_text: input split into {len(chunks)} chunk(s) "
          f"(long-input guard against the {max_length}-token generation cap)")

    translated_chunks = [
        translate_text(tokenizer, model, chunk, src_lang, tgt_lang, device, max_length)
        for chunk in chunks
    ]
    return " ".join(translated_chunks)