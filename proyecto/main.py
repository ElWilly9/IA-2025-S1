import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import pdfplumber
from langchain.schema import Document

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")  # Asegúrate de tener .env con GEMINI_API_KEY

# Directorio con PDFs
DOCS_DIR = "./data/"

# Text Processing
CHUNK_SIZE = 2000  
CHUNK_OVERLAP = 200  

# Paso 1: Cargar y chunkear PDFs
def load_and_chunk_pdfs():
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DOCS_DIR, filename)
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extraer texto
                    text = page.extract_text()
                    
                    # Extraer tablas
                    tables = page.extract_tables()
                    table_text = ""
                    for table in tables:
                        for row in table:
                            # Filtrar valores None y convertir a string
                            row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                            table_text += row_text + "\n"
                    
                    # Combinar texto y tablas
                    combined_text = f"{text}\n\nTablas:\n{table_text}"
                    
                    # Crear documento con metadatos
                    doc = Document(
                        page_content=combined_text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1
                        }
                    )
                    documents.append(doc)
    
    # Dividir documentos en fragmentos
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separator="\n"  # Dividir por líneas para mantener coherencia
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Paso 2: Crear base de datos vectorial
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="multi-qa-distilbert-cos-v1")
    vector_store = Chroma.from_documents(chunks, embeddings, collection_name="bajaj_boxer")
    return vector_store

# Paso 3: Configurar el sistema RAG
def setup_rag(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2
    )
    
    # Prompt personalizado para tono natural y cordial
    prompt_template = """
    Eres un asistente experto en motocicletas Bajaj Boxer CT100 KS. 
    Tu trabajo es ayudar a los usuarios con información técnica, mantenimiento y operación de esta motocicleta específica.

    CONTEXTO RELEVANTE DE LA DOCUMENTACIÓN:
    {context}

    PREGUNTA DEL USUARIO: {question}

    INSTRUCCIONES PARA TU RESPUESTA:
    - Responde ÚNICAMENTE sobre la Bajaj Boxer CT100 KS
    - Usa un tono cordial, natural y profesional en español
    - Basa tu respuesta en la información del contexto proporcionado
    - Si la información está en una tabla, incluye los valores específicos
    - Si la información no está en el contexto, dilo claramente
    - Proporciona respuestas prácticas y útiles para el usuario
    - Incluye detalles técnicos relevantes cuando sea apropiado
    - Si mencionas especificaciones técnicas, cita los valores exactos
    - Mantén la respuesta completa pero concisa

    RESPUESTA:
    """ 
    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return rag_chain

# Paso 4: Ciclo interactivo en consola
def main():
    print("Cargando documentos...")
    chunks = load_and_chunk_pdfs()
    print(f"Se cargaron {len(chunks)} fragmentos de documentos.")
    print_document_samples(chunks)  # Agregar esta línea
    
    print("Creando base de datos vectorial...")
    vector_store = create_vector_store(chunks)
    
    print("Configurando sistema RAG...")
    rag_chain = setup_rag(vector_store)
    
    print("\n¡Asistente virtual para Bajaj Boxer CT100 KS listo!")
    print("Escribe tu pregunta (o 'salir' para terminar):")
    
    while True:
        query = input("🧠> ")
        if query.lower() == ["salir", "quit", "exit"]:
            break
        
        result = rag_chain.invoke({"query": query})
        print("\nRespuesta 🤖:", result["result"])
        #print("\nFuentes:")
        #for doc in result["source_documents"]:
        #    print(f"- {doc.metadata.get('source', 'Desconocido')}: {doc.page_content[:100]}...")
    
    print("¡Gracias por usar el asistente!")

def print_document_samples(chunks):
    print("\nMuestras de documentos cargados:")
    for i, chunk in enumerate(chunks[:3]):  # Mostrar primeros 3 chunks
        print(f"\nChunk {i+1}:")
        print(f"Fuente: {chunk.metadata.get('source')}")
        print(f"Página: {chunk.metadata.get('page')}")
        print("Contenido:")
        print(chunk.page_content[:200] + "...")  # Primeros 200 caracteres

if __name__ == "__main__":
    main()