from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import answer
 
app = FastAPI(title="Sales & Marketing RAG Assistant")
 
class Query(BaseModel):
    question: str
    top_n: int = 6
 
@app.get("/health")
def health(): return {"status": "ok"}
 
@app.post("/ask")
def ask(q: Query):
    return answer(q.question, top_n=q.top_n)
