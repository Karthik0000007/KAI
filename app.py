from core.stt import record_audio, transcribe_audio
from core.llm import get_response
from core.tts import speak_text

def main():
    print("🎙️ Speak to KAI in English or Japanese...")
    record_audio(duration=5)

    user_input, language_code = transcribe_audio("input.wav")

    print(f"🗣️ You said: {user_input}")
    print(f"🌐 Language code: {language_code}")

    reply = get_response(user_input)
    print(f"🤖 KAI replies: {reply}")

    # Speak using detected language
    speak_text(reply, language=language_code)

if __name__ == "__main__":
    main()