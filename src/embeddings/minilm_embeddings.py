from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import MINILM_MODEL, EMBEDDING_DEVICE


def get_minilm_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a MiniLM embeddings object (Hugging Face).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=MINILM_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )
    return embeddings