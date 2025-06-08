from google import genai
from .vector_store import get_vectorstore
from . import config

retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})

def responder_pregunta(pregunta: str) -> str:
    documentos = retriever.get_relevant_documents(pregunta)
    contexto = "\n".join([doc.page_content for doc in documentos])
    prompt = f"Contesta en español y de forma cordial con base en esta información:\n{contexto}\n\nPregunta: {pregunta}"

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text