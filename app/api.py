from fastapi import FastAPI
from pydantic import BaseModel

from .rag_pipeline import build_rag_chain

app = FastAPI(title="RAG API")

# Build RAG system once (LLM + retriever + prompt)
rag_chain = build_rag_chain()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    result = rag_chain(req.question)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

@app.get("/")
def root():
    return {"message": "RAG API is running"}
