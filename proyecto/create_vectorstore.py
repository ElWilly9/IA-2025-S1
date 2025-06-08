from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from load_documents import load_and_chunk_pdf

def create_chroma_vectorstore(pdf_path, persist_directory="db"):
    chunks = load_and_chunk_pdf(pdf_path)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Base vectorial guardada en: {persist_directory}")
