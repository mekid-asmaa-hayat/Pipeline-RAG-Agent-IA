"""
Tests unitaires du pipeline RAG.
Couvre : ingestion, découpage, stats NumPy, recherche vectorielle.
"""
import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.ingestion.loader import split_documents, compute_chunk_stats, ingest_file
from app.config import settings


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Le traitement du signal est une discipline fondamentale en ingénierie. " * 20,
            metadata={"source": "test.txt", "page": 1}
        ),
        Document(
            page_content="Les algorithmes de détection radar utilisent la transformée de Fourier. " * 20,
            metadata={"source": "test.txt", "page": 2}
        ),
    ]

@pytest.fixture
def sample_txt_file():
    content = "Pipeline RAG pour le traitement de données documentaires.\n\n" * 50
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(content)
        yield f.name
    os.unlink(f.name)


# ── Tests ingestion ───────────────────────────────────────────────────────────

def test_split_documents_creates_chunks(sample_documents):
    chunks = split_documents(sample_documents)
    assert len(chunks) > 0
    assert all(isinstance(c, Document) for c in chunks)

def test_split_documents_respects_chunk_size(sample_documents):
    chunks = split_documents(sample_documents)
    for chunk in chunks:
        assert len(chunk.page_content) <= settings.chunk_size * 1.1  # tolérance 10%

def test_compute_chunk_stats_numpy(sample_documents):
    chunks = split_documents(sample_documents)
    stats = compute_chunk_stats(chunks)
    assert "total_chunks" in stats
    assert "mean_length" in stats
    assert "std_length" in stats
    assert stats["min_length"] <= stats["median_length"] <= stats["max_length"]
    assert stats["total_chunks"] == len(chunks)

def test_compute_chunk_stats_types(sample_documents):
    chunks = split_documents(sample_documents)
    stats = compute_chunk_stats(chunks)
    # Vérifie que les types NumPy sont bien convertis en types Python natifs
    assert isinstance(stats["total_chunks"], int)
    assert isinstance(stats["mean_length"], float)

def test_ingest_txt_file(sample_txt_file):
    with patch("app.ingestion.loader.load_document") as mock_load:
        mock_load.return_value = [
            Document(page_content="Contenu test. " * 100, metadata={"source": sample_txt_file})
        ]
        result = ingest_file(sample_txt_file)
        assert "chunks" in result
        assert "stats" in result
        assert result["stats"]["total_chunks"] > 0

def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Format non supporté"):
        from app.ingestion.loader import load_document
        load_document("document.docx")


# ── Tests NumPy ───────────────────────────────────────────────────────────────

def test_numpy_stats_single_chunk():
    chunks = [Document(page_content="Texte court.", metadata={})]
    stats = compute_chunk_stats(chunks)
    assert stats["total_chunks"] == 1
    assert stats["std_length"] == 0.0

def test_numpy_stats_variance():
    chunks = [
        Document(page_content="A" * 100, metadata={}),
        Document(page_content="B" * 500, metadata={}),
        Document(page_content="C" * 1000, metadata={}),
    ]
    stats = compute_chunk_stats(chunks)
    lengths = np.array([100, 500, 1000])
    assert abs(stats["mean_length"] - float(np.mean(lengths))) < 0.01
    assert abs(stats["std_length"] - float(np.std(lengths))) < 0.01
