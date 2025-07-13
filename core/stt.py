import whisper
import sounddevice as sd
import soundfile as sf
import os

model = whisper.load_model("base")  # or "small" / "medium" if GPU allows

def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print(f"[🎙️] Recording {duration} seconds of audio...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"[🎧] Saved audio to {filename}")
    return filename

def transcribe_audio(filename="input.wav"):
    print("[📝] Transcribing audio with Whisper...")
    
    # Step 1: Detect language first (Whisper can do this)
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # 🔍 Predict language
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    print(f"[🌐] Detected language: {detected_lang}")

    # Step 2: Transcribe with that language
    result = model.transcribe(filename, language=detected_lang)

    print(f"[🧠] Transcription: {result['text']}")
    return result["text"], detected_lang

