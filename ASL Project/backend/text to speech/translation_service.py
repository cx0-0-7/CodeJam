from google.cloud import translate

# Create a client
translate_client = translate.TranslationServiceClient()

def translate_to_english(text):
    if not text or text.strip() == "":
        return ""

    try:
        response = translate_client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "target_language_code": "en"
            }
        )
        # The API may return multiple translations; take the first
        return response.translations[0].translated_text

    except Exception as e:
        print(f"Translation Error: {e}")
        return text