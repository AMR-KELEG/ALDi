# https://mt.qcri.org/api/servers

from typing import Any
import requests
API_KEY = "API_KEY"



def find_translation(api_key: str, lang_pair: str, domain: str, text: str) -> str:
    """Translate the text using one of QCRI's Shaheen models.

    Args:
        api_key: API key acquired from https://mt.qcri.org/api/register
        lang_pair: ar-en or en-ar
        domain: The domain of the model from https://mt.qcri.org/api/servers
        text: The text to be translated

    Returns:
        A translation of the text.
    """
    url = f"https://mt.qcri.org/api/v1/translate?key={api_key}&langpair={lang_pair}&domain={domain}&text={text}"
    response = requests.get(url)
    return response.json()["translatedText"]


def backtranslate(arabic_text: str) -> dict[str, str | dict[str, Any]]:
    """Backtranslate text using QCRI's Shaheen models.

    Args:
        arabic_text: An input text to backtranslate.

    Returns:
        Back translated versions of the input text.
    """
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
