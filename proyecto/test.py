from google import genai
import os
from groq import Groq

query = (
    "¿Sabes acerca de la bocer ct100 ks? Responde en español, con un tono amigable, como si fueras un amigo contando algo interesante. "
    "No digas que eres una IA ni expliques cómo sabes la información. Solo responde directamente."
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=query,
)
print(response.text)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": query,
        }
    ],
    model="deepseek-r1-distill-llama-70b",
)

print(chat_completion.choices[0].message.content)


