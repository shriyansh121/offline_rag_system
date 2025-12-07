from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.config.settings import K, FETCH_K


def get_similarity_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    """
    Simple similarity-based retriever.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K},
    )
    return retriever


def get_mmr_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    """
    Max Marginal Relevance retriever (diverse results).
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": K, "fetch_k": FETCH_K},
    )
    return retriever
