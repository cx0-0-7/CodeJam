from google.cloud import translate

# Create a client
translate_client = translate.TranslationServiceClient()

def translate_text(text, target_language):
    """
    Translates text into the target language.
    
    Args:
        text (str): The text to translate.
        target_language (str): The language code to translate the text into.
    
    Returns:
        str: The translated text.
    """
    project_id = "your-project-id"  # Replace with your GCP project ID
    location = "global"
    
    parent = f"projects/{project_id}/locations/{location}"
    
    response = translate_client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "target_language_code": target_language,
        }
    )
    
    for translation in response.translations:
        return translation.translated_text