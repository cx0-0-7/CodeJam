#converting text to speech
from gtts import gTTS
import io
import os 

def convert_text_to_audio(text_to_speak, lang='en', save_path=None):
    """
    Converts text to speech using gTTS.
    
    Args:
        text_to_speak (str): The text to convert to audio.
        lang (str): Language code for TTS (default 'en').
        save_path (str, optional): If provided, saves audio to this file path.
    
    Returns:
        io.BytesIO: A BytesIO object containing the MP3 audio data, or None if an error occurs.
    """
    try:
        tts = gTTS(text=text_to_speak, lang=lang)
        
        # Save to in-memory buffer
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)  # Rewind buffer to start
        
        # Optional: save to disk
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(audio_fp.getbuffer())
        
        return audio_fp
    
    except Exception as e:
        print(f"TTS Error: {e}")
        return None
    

if __name__ == "__main__":
    # Ask for the text to convert
    user_text = input("Enter the text you want to convert to speech: ")
    
    # Ask for the desired file name
    file_name = input("Enter the file name for your MP3 (without extension): ")

    #Ask for desired language audio
    language = input("Enter the language (default is English, write 'es' for Spanish,"
                           "'fr' for French, 'de' for German, etc.")
    
    # Set the folder path
    folder = "MP3"
    
    # Full path for saving
    save_path = os.path.join("ASL Project/backend/text to speech", folder, f"{file_name}.mp3")
    
    # Convert and save
    audio_stream = convert_text_to_audio(user_text, lang = language, save_path=save_path)
    
    if audio_stream:
        print(f"TTS conversion successful! Your file is saved at: {os.path.abspath(save_path)}")