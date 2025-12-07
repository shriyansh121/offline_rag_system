import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

from src.config.settings import PDF_DIR


def load_pdfs(pdf_dir: str | None = None) -> List[Document]:
    """
    Load all PDFs from the given directory using PyPDFLoader.
    """
    target_dir = pdf_dir or PDF_DIR

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    loader = DirectoryLoader(
        target_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()
    return docs
