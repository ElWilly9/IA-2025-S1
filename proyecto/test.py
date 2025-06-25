from google import genai
import os
from groq import Groq

query = (
    "¿Sabes acerca de la boxer ct100 ks? Responde en español, con un tono amigable, como si fueras un amigo contando algo interesante. "
    "No digas que eres una IA ni expliques cómo sabes la información. Solo responde directamente."
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=query,
)
print("*****************************************************\n")
print("Modelo: Gemini 2.0 flash")
print(response.text)


models = ["llama-3.3-70b-versatile", "gemma2-9b-it", "deepseek-r1-distill-llama-70b"]
client2 = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

for model in models:
    print("\n*****************************************************\n")
    print("Modelo: ", model)
    chat_completion = client2.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model,
    )

    respuesta = chat_completion.choices[0].message.content
    print(respuesta)


