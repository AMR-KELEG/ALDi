# https://mt.qcri.org/api/servers

import requests
API_KEY = "API_KEY"


def find_translation(api_key, lang_pair, domain, text):
    url = f"https://mt.qcri.org/api/v1/translate?key={api_key}&langpair={lang_pair}&domain={domain}&text={text}"
    response = requests.get(url)
    return response.json()["translatedText"]


def backtranslate(arabic_text):
    #     translated_english = find_translation(API_KEY, lang_pair="ar-en", domain="dialectal", text=arabic_text)
    translated_english_general = find_translation(
        API_KEY, lang_pair="ar-en", domain="general-fast", text=arabic_text
    )
    translated_english_dialectal = find_translation(
        API_KEY, lang_pair="ar-en", domain="dialectal", text=arabic_text
    )
    translated_arabic_general = find_translation(
        API_KEY, lang_pair="en-ar", domain="general", text=translated_english_general
    )
    translated_arabic_dialectal = find_translation(
        API_KEY, lang_pair="en-ar", domain="general", text=translated_english_dialectal
    )
    return {
        "arabic": arabic_text,
        "translated_english": {
            "general": translated_english_general,
            "dialectal": translated_english_dialectal,
        },
        "backtranslated_arabic": {
            "general": translated_arabic_general,
            "dialectal": translated_arabic_dialectal,
        },
    }


if __name__ == "__main__":
    arabic_text = input("Enter an Arabic text:\n")
    print(backtranslate(arabic_text))
