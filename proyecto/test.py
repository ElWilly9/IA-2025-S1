from google import genai

key = "AIzaSyDHgW3Dqou631Yb2BQV1eHPzv2OCXUfIR0"
query = "conces acerca de la moto bajaj boxer ct100 KS?"

client = genai.Client(api_key=key)

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=query,
)
print(response.text)


