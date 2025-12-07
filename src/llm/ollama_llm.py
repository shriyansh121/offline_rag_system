from langchain_ollama import ChatOllama

from src.config.settings import LLM_MODEL, LLM_TEMPERATURE


def get_llm() -> ChatOllama:
    """
    Returns a ChatOllama instance with the configured model.
    Make sure you have Ollama installed and have run:
      ollama run gemma3:4b
    at least once.
    """
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )
    return llm
