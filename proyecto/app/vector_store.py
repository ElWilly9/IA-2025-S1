from langchain.vectorstores import Chroma
from langchain.embeddings import FakeEmbeddings
from . import config

embedding = FakeEmbeddings(size=768)

def get_vectorstore():
    return Chroma(persist_directory=config.CHROMA_DB_DIR, embedding_function=embedding)