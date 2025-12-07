import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config.settings import CHROMA_DIR
from src.embeddings.minilm_embeddings import get_minilm_embeddings


def _chroma_dir_has_content(persist_directory: str) -> bool:
    """
    Quick check to see if a Chroma directory likely has an existing index.
    """
    if not os.path.exists(persist_directory):
        return False
    # crude check: any files?
    return any(os.scandir(persist_directory))


def build_vectorstore_from_docs(
    docs: list[Document],
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Create a new Chroma vectorstore from documents and persist it.
    """
    persist_directory = persist_directory or CHROMA_DIR
    embeddings = get_minilm_embeddings()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_or_create_vectorstore(
    docs: Optional[list[Document]] = None,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Load an existing Chroma store if present, otherwise build a new one
    from provided docs.
    """
    persist_directory = persist_directory or CHROMA_DIR
    embeddings = get_minilm_embeddings()

    if _chroma_dir_has_content(persist_directory):
        # Load existing DB
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
    else:
        if docs is None:
            raise ValueError(
                "No existing Chroma DB found and no docs provided to build a new one."
            )
        vectorstore = build_vectorstore_from_docs(docs, persist_directory=persist_directory)

    return vectorstore
