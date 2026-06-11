# RAG Pipeline — Traitement de Données Documentaires

Pipeline RAG (Retrieval-Augmented Generation) modulaire en Python/FastAPI. Ingestion documentaire, vectorisation FAISS, chaînes LangChain multi-étapes avec OpenAI et Anthropic.

## Stack technique

- **Python 3.11** · **FastAPI** · **LangChain** · **NumPy**
- **FAISS** (vectorisation et indexation)
- **OpenAI** (GPT-4o-mini + text-embedding-3-small) · **Anthropic** (Claude Haiku)
- **Docker** · **Docker Compose** · **GitHub Actions CI/CD**

## Architecture

```
rag-pipeline/
├── app/
│   ├── ingestion/      # Chargement et découpage documentaire (PDF, TXT, MD)
│   ├── vectorstore/    # Vectorisation FAISS + recherche par similarité cosinus
│   ├── chains/         # Chaînes LangChain RAG et résumé (OpenAI + Anthropic)
│   └── main.py         # API FastAPI — 5 routes documentées
├── tests/              # Tests unitaires (ingestion, NumPy, découpage)
├── data/               # Index FAISS persistant
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Fonctionnement

1. **Ingestion** (`POST /ingest`) : upload PDF/TXT/MD → découpage RecursiveCharacterTextSplitter → vectorisation OpenAI Embeddings → index FAISS sauvegardé sur disque
2. **Requête RAG** (`POST /query`) : question → retrieval FAISS top-k → prompt LangChain → réponse LLM (OpenAI ou Anthropic au choix) + sources
3. **Recherche vectorielle** (`POST /search`) : similarité cosinus pure, scores normalisés NumPy
4. **Résumé** (`POST /summarize`) : chaîne LangChain dédiée sur texte libre

## Démarrage rapide

```bash
# 1. Configuration
cp .env.example .env
# Renseigner OPENAI_API_KEY et/ou ANTHROPIC_API_KEY dans .env

# 2. Lancement Docker
docker-compose up --build

# 3. API disponible
# http://localhost:8000/docs  (Swagger UI)
# http://localhost:8000/health
```

## Sans Docker

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Statut API + configuration |
| POST | `/ingest` | Ingestion document (PDF/TXT/MD) |
| POST | `/query` | Interrogation RAG (OpenAI ou Anthropic) |
| POST | `/search` | Recherche vectorielle FAISS |
| POST | `/summarize` | Résumé LangChain |
| GET | `/stats` | Statistiques index FAISS |

## Exemple d'utilisation

```bash
# Ingérer un document
curl -X POST http://localhost:8000/ingest \
  -F "file=@rapport.pdf"

# Interroger avec OpenAI
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les points clés du rapport ?", "provider": "openai"}'

# Interroger avec Anthropic
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Résume les conclusions.", "provider": "anthropic"}'
```

## Tests

```bash
pytest tests/ -v --cov=app
```

## Points techniques notables

- **Calcul NumPy** : statistiques sur les chunks (mean, std, median, distribution des longueurs) + normalisation des scores de similarité
- **Chaînes multi-étapes** : `RunnableParallel` LangChain pour paralléliser retrieval et passage de la question
- **Switch provider** : OpenAI / Anthropic sélectionnable par requête sans redémarrage
- **Persistance FAISS** : index sauvegardé sur disque, rechargé à chaud
- **CI/CD** : GitHub Actions — tests + build Docker + smoke test container
