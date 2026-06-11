"""
Module d'ingestion documentaire.
Supporte : PDF, TXT, Markdown.
Découpe les documents en chunks via RecursiveCharacterTextSplitter.
"""
import numpy as np
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.config import settings


def load_document(file_path: str) -> List[Document]:
    """Charge un document selon son extension."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Format non supporté : {ext}. Formats acceptés : .pdf, .txt, .md")

    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Découpe les documents en chunks.
    Utilise RecursiveCharacterTextSplitter pour respecter la structure naturelle du texte.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def compute_chunk_stats(chunks: List[Document]) -> dict:
    """Calcule des statistiques NumPy sur les chunks (longueurs, distribution)."""
    lengths = np.array([len(chunk.page_content) for chunk in chunks])
    return {
        "total_chunks": int(len(chunks)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "median_length": float(np.median(lengths)),
    }


def ingest_file(file_path: str) -> dict:
    """Pipeline complet d'ingestion : chargement → découpage → statistiques."""
    documents = load_document(file_path)
    chunks = split_documents(documents)
    stats = compute_chunk_stats(chunks)
    return {"chunks": chunks, "stats": stats, "source": file_path}
