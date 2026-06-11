"""
Chaînes LangChain multi-étapes pour la génération RAG.
Supporte OpenAI (GPT-4o-mini) et Anthropic (Claude Haiku).
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from app.config import settings
from app.vectorstore.index import load_index


RAG_PROMPT = ChatPromptTemplate.from_template("""
Tu es un assistant expert en analyse documentaire.
Utilise uniquement le contexte fourni pour répondre à la question.
Si la réponse n'est pas dans le contexte, indique-le clairement.

Contexte :
{context}

Question : {question}

Réponse structurée et précise :
""")

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
Résume les points clés suivants de manière concise et structurée en français :

{context}

Résumé en 3-5 points clés :
""")


def get_llm(provider: str = None):
    """Retourne le LLM selon le provider configuré."""
    provider = provider or settings.llm_provider
    if provider == "anthropic":
        return ChatAnthropic(
            model=settings.anthropic_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.2,
    )


def format_context(docs: List[Document]) -> str:
    """Formate les documents récupérés en contexte texte."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'inconnu')} | Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(provider: str = None):
    """
    Construit la chaîne RAG complète :
    retriever → format_context → prompt → LLM → parser
    """
    vectorstore = load_index()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k_results},
    )
    llm = get_llm(provider)

    chain = (
        RunnableParallel(
            context=(retriever | (lambda docs: format_context(docs))),
            question=RunnablePassthrough(),
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def build_summary_chain(provider: str = None):
    """Chaîne de résumé : agrège les chunks pertinents et génère un résumé."""
    llm = get_llm(provider)
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    return chain


def run_rag_query(question: str, provider: str = None) -> dict:
    """Exécute une requête RAG complète et retourne la réponse avec les sources."""
    vectorstore = load_index()
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k_results})
    docs = retriever.invoke(question)

    chain = build_rag_chain(provider)
    answer = chain.invoke(question)

    sources = [
        {
            "source": doc.metadata.get("source", "inconnu"),
            "page": doc.metadata.get("page", "?"),
            "excerpt": doc.page_content[:200] + "...",
        }
        for doc in docs
    ]

    return {"answer": answer, "sources": sources, "provider": provider or settings.llm_provider}


def run_summary(context_text: str, provider: str = None) -> str:
    """Génère un résumé à partir d'un texte de contexte."""
    chain = build_summary_chain(provider)
    return chain.invoke({"context": context_text})
