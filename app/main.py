"""
Pipeline RAG — API FastAPI
Routes :
  POST /ingest       — Ingestion d'un document (PDF/TXT/MD)
  POST /query        — Interrogation RAG
  POST /search       — Recherche vectorielle pure
  POST /summarize    — Résumé de document
  GET  /health       — Santé de l'API
  GET  /stats        — Statistiques de l'index
"""
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.ingestion.loader import ingest_file
from app.vectorstore.index import build_index, similarity_search, load_index
from app.chains.rag_chain import run_rag_query, run_summary
from app.config import settings

app = FastAPI(
    title="Pipeline RAG — Traitement de données documentaires",
    description=(
        "Pipeline RAG modulaire : ingestion documentaire, vectorisation FAISS, "
        "chaînes LangChain multi-étapes avec OpenAI et Anthropic."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    provider: Optional[str] = None  # "openai" | "anthropic"

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

class SummarizeRequest(BaseModel):
    text: str
    provider: Optional[str] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Système"])
def health():
    """Vérifie que l'API est opérationnelle."""
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
    }


@app.post("/ingest", tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingère un document (PDF, TXT, MD) :
    chargement → découpage → vectorisation → indexation FAISS.
    """
    allowed = {".pdf", ".txt", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Format non supporté : {ext}. Acceptés : {allowed}")

    # Sauvegarde temporaire du fichier uploadé
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingest_file(tmp_path)
        chunks = result["chunks"]

        # Mise à jour du métadonnées source avec le nom original
        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        build_index(chunks)

        return {
            "message": f"Document '{file.filename}' ingéré avec succès.",
            "filename": file.filename,
            "chunks_created": result["stats"]["total_chunks"],
            "stats": result["stats"],
        }
    finally:
        os.unlink(tmp_path)


@app.post("/query", tags=["RAG"])
def query_documents(request: QueryRequest):
    """
    Interroge la base documentaire via RAG.
    Sélectionne le provider LLM (OpenAI ou Anthropic) à la volée.
    """
    if not request.question.strip():
        raise HTTPException(400, "La question ne peut pas être vide.")
    try:
        result = run_rag_query(request.question, provider=request.provider)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Erreur lors de la génération : {str(e)}")


@app.post("/search", tags=["Vectorstore"])
def search_documents(request: SearchRequest):
    """
    Recherche vectorielle pure dans l'index FAISS.
    Retourne les k chunks les plus similaires avec leurs scores normalisés.
    """
    if not request.query.strip():
        raise HTTPException(400, "La requête ne peut pas être vide.")
    try:
        results = similarity_search(request.query, k=request.k)
        return {"query": request.query, "results": results}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/summarize", tags=["RAG"])
def summarize_text(request: SummarizeRequest):
    """Génère un résumé structuré du texte fourni via LangChain."""
    if not request.text.strip():
        raise HTTPException(400, "Le texte ne peut pas être vide.")
    try:
        summary = run_summary(request.text, provider=request.provider)
        return {
            "summary": summary,
            "provider": request.provider or settings.llm_provider,
        }
    except Exception as e:
        raise HTTPException(500, f"Erreur lors du résumé : {str(e)}")


@app.get("/stats", tags=["Système"])
def index_stats():
    """Retourne des informations sur l'index vectoriel courant."""
    try:
        vs = load_index()
        n_vectors = vs.index.ntotal
        return {
            "index_loaded": True,
            "total_vectors": n_vectors,
            "embedding_model": settings.embedding_model,
            "index_path": "data/faiss_index",
        }
    except FileNotFoundError:
        return {"index_loaded": False, "total_vectors": 0}
