from typing import List, Dict, Any
from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """
    Join page_content from docs for context.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract source info (file name, page) from Document.metadata.
    """
    sources = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", None)
        sources.append(
            {
                "source": source,
                "page": page,
                "snippet": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            }
        )
    return sources


def chat_history_to_str(chat_history):
    """
    Convert chat history (list of dicts OR LangChain messages) into a string.
    Supports both formats.
    """
    lines = []
    for msg in chat_history:
        
        # Case 1 — LangChain message objects (AIMessage, HumanMessage)
        if hasattr(msg, "content"):
            role = getattr(msg, "type", getattr(msg, "role", ""))
            prefix = "User" if role == "human" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        
        # Case 2 — Our custom dict messages
        elif isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")

        else:
            # fallback
            lines.append(str(msg))

    return "\n".join(lines)
