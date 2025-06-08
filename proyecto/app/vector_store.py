from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from . import config

embedding = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vectorstore():
    return Chroma(persist_directory=config.CHROMA_DB_DIR, embedding_function=embedding)