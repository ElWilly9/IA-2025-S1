import os
import glob
from pathlib import Path
import chromadb
from chromadb.config import Settings
from google import genai
import PyPDF2
from typing import List, Dict
import uuid

class BajajRAGAssistant:
    def __init__(self, api_key: str, docs_folder: str = "documentos"):
        """
        Inicializa el asistente RAG para Bajaj Boxer CT100 KS
        
        Args:
            api_key: Clave API de Google Gemini
            docs_folder: Carpeta con documentos PDF
        """
        self.api_key = "AIzaSyDHgW3Dqou631Yb2BQV1eHPzv2OCXUfIR0"
        self.docs_folder = "./data/"
        self.client = genai.Client(api_key=api_key)
        
        # Configurar ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False 
        ))
        
        # Crear o obtener colección
        self.collection = self.chroma_client.get_or_create_collection(
            name="bajaj_boxer_ct100_docs",
            metadata={"description": "Documentos de Bajaj Boxer CT100 KS"}
        )
        
        print(f"✅ Asistente RAG inicializado")
        print(f"📁 Carpeta de documentos: {docs_folder}")
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrae texto de un archivo PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"❌ Error al leer {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Divide el texto en chunks con overlap
        
        Args:
            text: Texto a dividir
            chunk_size: Tamaño de cada chunk
            overlap: Solapamiento entre chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
                
        return chunks
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Genera embeddings usando Gemini"""
        try:
            response = self.client.models.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            return response.embedding
        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            return []
    
    def load_documents(self):
        """Carga y procesa todos los PDFs de la carpeta"""
        pdf_files = glob.glob(os.path.join(self.docs_folder, "*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No se encontraron archivos PDF en {self.docs_folder}")
            return
        
        print(f"📚 Procesando {len(pdf_files)} documentos...")
        
        for pdf_file in pdf_files:
            print(f"📄 Procesando: {os.path.basename(pdf_file)}")
            
            # Extraer texto
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                continue
                
            # Dividir en chunks
            chunks = self.chunk_text(text)
            print(f"   ✂️  Dividido en {len(chunks)} chunks")
            
            # Procesar cada chunk
            for i, chunk in enumerate(chunks):
                # Generar embedding
                embedding = self.generate_embeddings(chunk)
                if not embedding:
                    continue
                
                # Almacenar en ChromaDB
                chunk_id = f"{os.path.basename(pdf_file)}_chunk_{i}"
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "source": os.path.basename(pdf_file),
                        "chunk_id": i,
                        "file_path": pdf_file
                    }],
                    ids=[chunk_id]
                )
            
            print(f"   ✅ {os.path.basename(pdf_file)} procesado correctamente")
        
        print(f"🎉 Todos los documentos cargados exitosamente!")
    
    def search_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """Busca documentos relevantes usando embeddings"""
        try:
            # Generar embedding de la consulta
            query_embedding = self.generate_embeddings(query)
            if not query_embedding:
                return []
            
            # Buscar en ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Formatear resultados
            documents = []
            for i in range(len(results['documents'][0])):
                documents.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0
                })
            
            return documents
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Genera respuesta usando Gemini con contexto"""
        # Construir contexto
        context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Prompt especializado para Bajaj Boxer CT100 KS
        prompt = f"""
Eres un asistente experto en motocicletas Bajaj Boxer CT100 KS. 
Responde de manera cordial y natural en español.

CONTEXTO RELEVANTE:
{context}

PREGUNTA DEL USUARIO: {query}

INSTRUCCIONES:
- Responde específicamente sobre la Bajaj Boxer CT100 KS
- Usa un tono cordial y natural
- Si la información no está en el contexto, dilo claramente
- Da respuestas prácticas y útiles
- Mantén la respuesta concisa pero completa

RESPUESTA:
"""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"❌ Error generando respuesta: {e}"
    
    def ask(self, query: str) -> str:
        """Función principal para hacer preguntas"""
        print(f"\n🤔 Pregunta: {query}")
        print("🔍 Buscando información relevante...")
        
        # Buscar documentos relevantes
        relevant_docs = self.search_documents(query)
        
        if not relevant_docs:
            return "❌ No encontré información relevante para tu pregunta."
        
        print(f"📋 Encontrados {len(relevant_docs)} documentos relevantes")
        
        # Generar respuesta
        answer = self.generate_answer(query, relevant_docs)
        return answer
    
    def interactive_mode(self):
        """Modo interactivo por consola"""
        print("\n" + "="*50)
        print("🏍️  ASISTENTE BAJAJ BOXER CT100 KS")
        print("="*50)
        print("Escribe 'salir' para terminar")
        print("Escribe 'recargar' para volver a cargar documentos")
        print("-"*50)
        
        while True:
            try:
                query = input("\n💬 Tu pregunta: ").strip()
                
                if query.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break
                
                if query.lower() == 'recargar':
                    self.load_documents()
                    continue
                
                if not query:
                    continue
                
                # Procesar pregunta
                answer = self.ask(query)
                print(f"\n🤖 Respuesta:\n{answer}")
                
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    # Configuración
    API_KEY = "AIzaSyDHgW3Dqou631Yb2BQV1eHPzv2OCXUfIR0"  # Tu clave API
    DOCS_FOLDER = "documentos"  # Carpeta con PDFs
    
    # Crear carpeta si no existe
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    
    # Inicializar asistente
    assistant = BajajRAGAssistant(API_KEY, DOCS_FOLDER)
    
    # Cargar documentos
    assistant.load_documents()
    
    # Modo interactivo
    assistant.interactive_mode()

if __name__ == "__main__":
    main()