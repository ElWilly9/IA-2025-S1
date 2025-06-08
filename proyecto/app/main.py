from fastapi import FastAPI
from .api import router
from .rag_pipeline import responder_pregunta, chain
from fastapi.responses import JSONResponse


app = FastAPI(title="Asistente Virtual Bajaj CT100 KS")
app.include_router(router)

@app.get("/")
def read_root():
    return {"mensaje": "Bienvenido a la API del Asistente Virtual Bajaj CT100 KS"}

@app.get("/consulta")
def consulta(pregunta: str):
    try:
        return {"respuesta": responder_pregunta(pregunta)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Ocurrió un error interno", "detalle": str(e)},
        )
