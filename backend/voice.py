
from gtts import gTTS
import os

def generate_voice(text, language, filename="output.mp3"):
    tts = gTTS(text=text, lang=language)
    tts.save(filename)
    return filename
