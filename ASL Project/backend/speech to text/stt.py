import speech_recognition as sr

def record_and_transcribe():
    recognizer = sr.Recognizer()

    # Use microphone as input
    with sr.Microphone() as source:
        print("Adjusting for background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print("🎤 Start speaking now...")
        audio = recognizer.listen(source)

    print("Processing audio...")

    try:
        # Use Google Web Speech API (free, no API key needed)
        text = recognizer.recognize_google(audio)
        print("📝 Transcription:")
        print(text)
        return text

    except sr.UnknownValueError:
        print("❌ Speech was not understood.")
    except sr.RequestError as e:
        print(f"❌ Could not request results; {e}")

#Run the function
if __name__ == "__main__":
    record_and_transcribe()