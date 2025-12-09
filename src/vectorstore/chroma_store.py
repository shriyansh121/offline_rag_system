import os
from typing import Optional, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config.settings import CHROMA_DIR, PDF_DIR
from src.embeddings.minilm_embeddings import get_minilm_embeddings
from src.loaders.pdf_loader import load_pdfs
from src.splitter.text_splitter import split_docs


def _chroma_dir_has_content(path: str) -> bool:
    """Return True if Chroma DB directory contains files."""
    return os.path.exists(path) and any(os.scandir(path))


def _get_existing_sources(vectorstore: Chroma) -> set:
    """
    Extract list of existing source file paths stored in Chroma metadata.
    Used to detect which PDFs are already embedded.
    """
    existing = set()
    try:
        collection = vectorstore._collection.get(include=["metadatas"], limit=9999999)
        for meta in collection["metadatas"]:
            if meta and "source" in meta:
                existing.add(meta["source"])
    except Exception as e:
        print("[WARN] Could not read existing metadata:", e)
    return existing


def _get_all_pdf_paths(pdf_dir: str = PDF_DIR) -> List[str]:
    """Return absolute paths for all PDFs in the directory."""
    pdfs = []
    if not os.path.exists(pdf_dir):
        return pdfs
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdfs.append(os.path.join(pdf_dir, filename))
    return pdfs


def _get_new_pdfs(existing_sources: set, pdf_paths: List[str]) -> List[str]:
    """Return only PDFs not yet embedded."""
    return [p for p in pdf_paths if p not in existing_sources]


def _embed_and_add_new_pdfs(vectorstore: Chroma, new_pdf_paths: List[str]):
    """Load, split, embed, and add new PDFs into existing Chroma DB."""
    if not new_pdf_paths:
        return

    print(f"[INFO] Found {len(new_pdf_paths)} new PDFs → embedding...")

    # Load ALL docs then filter to only new PDFs (DirectoryLoader adds 'source' metadata)
    docs = load_pdfs(PDF_DIR)
    docs = [d for d in docs if d.metadata.get("source") in new_pdf_paths]

    if not docs:
        print("[INFO] No matching docs found for new PDFs (metadata/path mismatch or unreadable PDFs).")
        return

    chunks = split_docs(docs)
    if not chunks:
        print("[INFO] No text chunks created from new PDFs (maybe scanned/image-only docs). Skipping add.")
        return

    vectorstore.add_documents(chunks)
    print(f"[INFO] Added {len(chunks)} new chunks into vectorstore.")


def load_or_update_vectorstore(
    persist_directory: Optional[str] = None,
    pdf_dir: Optional[str] = None,
) -> Chroma:
    """
    Load existing Chroma DB if available.
    If new PDFs are detected, embed ONLY new PDFs and update DB.
    If DB does not exist, create a new one from all PDFs.
    """
    persist_directory = persist_directory or CHROMA_DIR
    pdf_dir = pdf_dir or PDF_DIR

    embeddings = get_minilm_embeddings()
    pdf_paths = _get_all_pdf_paths(pdf_dir)

    # Case 1: DB does not exist → build fresh DB
    if not _chroma_dir_has_content(persist_directory):
        print("[INFO] No existing Chroma DB found → creating new one.")

        docs = load_pdfs(pdf_dir)
        if not docs:
            print("[WARN] No PDFs found to index. Creating empty vectorstore.")
            # Empty collection – still usable, just returns nothing
            return Chroma(
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )

        splits = split_docs(docs)
        if not splits:
            print("[WARN] No chunks created from PDFs. Creating empty vectorstore.")
            return Chroma(
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        print(f"[INFO] Built new vectorstore with {len(splits)} chunks.")
        return vectorstore

    # Case 2: DB exists → load it
    print("[INFO] Loading existing Chroma DB...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Detect & index new PDFs
    existing_sources = _get_existing_sources(vectorstore)
    new_pdf_paths = _get_new_pdfs(existing_sources, pdf_paths)

    if new_pdf_paths:
        print(f"[INFO] New PDFs detected: {new_pdf_paths}")
        _embed_and_add_new_pdfs(vectorstore, new_pdf_paths)
    else:
        print("[INFO] No new PDFs found.")

    return vectorstore
