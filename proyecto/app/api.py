from fastapi import APIRouter
from pydantic import BaseModel
from .rag_pipeline import responder_pregunta

router = APIRouter()

class Consulta(BaseModel):
    pregunta: str

@router.post("/consulta")
def consultar(entrada: Consulta):
    respuesta = responder_pregunta(entrada.pregunta)
    return {"respuesta": respuesta}