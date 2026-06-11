"""
Module de vectorisation et indexation FAISS.
Gère la création, la sauvegarde et le chargement de l'index vectoriel.
"""
import os
import numpy as np
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings

INDEX_PATH = "data/faiss_index"


def get_embeddings():
    """Retourne le modèle d'embeddings OpenAI."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def build_index(chunks: List[Document]) -> FAISS:
    """
    Vectorise les chunks et construit l'index FAISS.
    Sauvegarde l'index sur disque pour réutilisation.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore


def load_index() -> FAISS:
    """Charge l'index FAISS depuis le disque."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "Index FAISS introuvable. Veuillez d'abord ingérer des documents via POST /ingest."
        )
    embeddings = get_embeddings()
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def similarity_search(query: str, k: int = None) -> List[Tuple[Document, float]]:
    """
    Recherche les k chunks les plus similaires à la requête.
    Retourne les documents avec leur score de similarité cosinus.
    """
    k = k or settings.top_k_results
    vectorstore = load_index()
    results = vectorstore.similarity_search_with_score(query, k=k)

    # Normalisation NumPy des scores pour affichage
    scores = np.array([score for _, score in results])
    normalized = 1 - (scores / (scores.max() + 1e-9))

    return [
        {"content": doc.page_content, "metadata": doc.metadata, "score": float(norm)}
        for (doc, _), norm in zip(results, normalized)
    ]
