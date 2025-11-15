import requests

def translate_to_english():
    print("🌍 TEXT TRANSLATOR TO ENGLISH")
    print("------------------------------------")

    text = input("Enter the text you want to translate: ").strip()
    original_lang = input("Enter the language code (ex: fr, es, ar): ").strip()

    url = "https://libretranslate.com/translate"

    payload = {
        "q": text,
        "source": original_lang,
        "target": "en",
        "format": "text",
        "api_key": ""   # Leave empty for now
    }

    try:
        response = requests.post(url, json=payload)
        print("Status code:", response.status_code)
        print("Response:", response.text)

        response.raise_for_status()  
        translated = response.json().get("translatedText", "")

        print("\n✅ TRANSLATION SUCCESSFUL!")
        print("Translated text:", translated)
        return translated

    except Exception as e:
        print("\n❌ ERROR: Could not translate text.")
        print("Reason:", str(e))
        return None


translate_to_english()

