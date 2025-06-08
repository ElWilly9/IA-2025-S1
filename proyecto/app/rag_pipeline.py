from google import genai
from .vector_store import get_vectorstore
from . import config

retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})

def responder_pregunta(pregunta: str) -> str:
    documentos = retriever.get_relevant_documents(pregunta)
    contexto = "\n".join([doc.page_content for doc in documentos])
    prompt = f"Eres un asistente virtual de conseccionario de Bajaj tu funcion es ayudarme a responder preguntas sobre la moto BAJAJ Boxer CT100 KS. Contesta en español y de forma cordial con base en esta información:\n{contexto}\n\nPregunta: {pregunta}\n Si no sabes la respuesta, di que no lo sabes, no inventes información, solo di que no lo sabes."

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text