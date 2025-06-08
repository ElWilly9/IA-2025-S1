import speech_recognition as sr
import pyttsx3 #Text-to-Speech
import os
import time

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Habla ahora. La grabación terminará cuando te quedes en silencio...")
    # Ajusta el umbral de energía para detectar silencio automáticamente
    r.adjust_for_ambient_noise(source, duration=0.6)
    audio = r.listen(source, timeout=None, phrase_time_limit=None)
    # Guardar el audio en un archivo WAV
    with open("proyecto\\query.wav", "wb") as f:
        f.write(audio.get_wav_data())
    try:    
        print("Espere un momento, procesando...")
        text = r.recognize_google(audio, language='es-ES')
        print("Texto reconocido:", text)
        engine = pyttsx3.init()
        engine.setProperty('rate', 120)  # Velocidad de habla
        engine.setProperty('volume', 1)  # Volumen (0.0 a 1.0)
        engine.say(text)
        engine.save_to_file(text, "proyecto\\answer.mp3")  # Guardar el texto a voz en un archivo MP3
        engine.runAndWait()
    except:
        print("Lo siento, no pude entender el audio.")