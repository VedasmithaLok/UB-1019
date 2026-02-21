
from googletrans import Translator

def translate_text(text, language):
    translator = Translator()
    translated = translator.translate(text, dest=language)
    return translated.text
