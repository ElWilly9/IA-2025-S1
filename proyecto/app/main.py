from fastapi import FastAPI
from .api import router

app = FastAPI(title="Asistente Virtual Bajaj CT100 KS")
app.include_router(router)