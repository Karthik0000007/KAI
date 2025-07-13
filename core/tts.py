import os
import json
import torch
import requests
from TTS.api import TTS
from TTS.utils.radam import RAdam
from collections import defaultdict
from langdetect import detect
from fugashi import Tagger

tagger = Tagger()

def clean_japanese_text(text):
    """
    Tokenizes and cleans Japanese text for better TTS clarity.
    """
    tokens = [word.surface for word in tagger(text)]
    return "".join(tokens)

# Coqui: preload once if needed
COQUI_MODEL_NAME =  "tts_models/en/jenny/jenny"#"tts_models/en/ljspeech/tacotron2-DDC"
coqui_tts = TTS(model_name=COQUI_MODEL_NAME)
coqui_tts.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Register safe globals for unpickling Coqui TTS models
torch.serialization.add_safe_globals([RAdam, defaultdict, dict])

# VOICEVOX: speaker config (Tsumugi)
VOICEVOX_URL = "http://127.0.0.1:50021"
TSUMUGI_SPEAKER_ID = 3

def detect_language(text):
    try:
        lang = detect(text)
        if lang.startswith("ja"):
            return "ja"
        elif lang.startswith("en"):
            return "en"
        else:
            return "en"  # fallback
    except:
        return "en"

def speak_text(text, language=None, filename="kai_response.wav"):
    """
    Synthesizes speech with language auto-detection if not specified.
    """
    if language is None:
        language = detect_language(text)

    # print(f"[TTS] Detected language: {language.upper()}")
    # print(f"[TTS] Speaking: {text}")

    if language == "ja":
        try:
            text = clean_japanese_text(text)
            query = requests.post(
                f"{VOICEVOX_URL}/audio_query",
                params={"text": text, "speaker": TSUMUGI_SPEAKER_ID}
            )
            synthesis = requests.post(
                f"{VOICEVOX_URL}/synthesis",
                headers={"Content-Type": "application/json"},
                params={"speaker": TSUMUGI_SPEAKER_ID},
                data=json.dumps(query.json())
            )
            with open(filename, "wb") as f:
                f.write(synthesis.content)

        except Exception as e:
            print(f"[TTS][JA] ❌ Error using VOICEVOX: {e}")
            return

    elif language == "en":
        try:
            coqui_tts.tts_to_file(text=text, file_path=filename)
        except Exception as e:
            print(f"[TTS][EN] ❌ Error using Coqui TTS: {e}")
            return
    else:
        print(f"[TTS] ❌ Unsupported language: {language}")
        return

    # Play output
    try:
        if os.name == "nt":
            os.system(f"start {filename}")
        elif os.name == "posix":
            os.system(f"aplay {filename}")
        else:
            print(f"[TTS] 🔊 Output saved to {filename}")
    except Exception as e:
        print(f"[TTS] ❌ Error playing audio: {e}")