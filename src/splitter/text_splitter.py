from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a RecursiveCharacterTextSplitter configured for MiniLM.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter


def split_docs(docs: List[Document]) -> List[Document]:
    splitter = get_text_splitter()
    return splitter.split_documents(docs)
