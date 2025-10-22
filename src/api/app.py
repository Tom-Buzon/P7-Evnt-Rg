# src/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.scripts.build_index import main as rebuild_index
from src.scripts.query_rag import answer

app = FastAPI(title="Puls-Events RAG API", version="0.1.0")

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod: restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskPayload(BaseModel):
    question: str
    k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(payload: AskPayload):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    res = answer(payload.question, k=payload.k)
    return res

@app.post("/rebuild")
def rebuild():
    try:
        rebuild_index()
        return {"status": "ok", "message": "Index rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
