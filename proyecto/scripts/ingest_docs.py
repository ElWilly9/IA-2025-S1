import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from app import config

DATA_DIR = "data/raw_docs"

all_docs = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        docs = loader.load()
        all_docs.extend(docs)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = splitter.split_documents(all_docs)

embedding = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Cambiar por tu modelo de embeddings real si deseas

vectordb = Chroma.from_documents(
    documents=docs_split,
    embedding=embedding,
    persist_directory=config.CHROMA_DB_DIR
)

vectordb.persist()
print("Base vectorial creada y guardada")