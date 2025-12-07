from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from src.utils.helpers import format_docs, chat_history_to_str
from src.config.settings import APP_NAME


def build_rag_chain(
    retriever,
    llm,
):
    """
    Build LCEL RAG chain WITHOUT memory built-in.
    """

    prompt = ChatPromptTemplate.from_template(
        (
            "You are a helpful assistant for an offline PDF knowledge system.\n"
            "Use ONLY the Context + Chat History.\n"
            "If unsure, say you don't know.\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    chain = (
        {
            # IMPORTANT FIX:
            # Only pass "question" to retriever. Not the entire dict.
            "context": RunnablePassthrough().pick("question") | retriever | format_docs,
            "question": RunnablePassthrough().pick("question"),
            "chat_history": RunnablePassthrough().pick("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def run_rag_with_memory(
    question: str,
    retriever,
    llm,
    memory,
) -> Dict[str, Any]:

    mem_vars = memory.load_memory_variables({})
    chat_history = mem_vars.get("chat_history", [])

    rag_chain = build_rag_chain(retriever, llm)

    chat_history_str = chat_history_to_str(chat_history)

    # retrieve documents directly for UI
    retrieved_docs: List[Document] = retriever.invoke(question)

    answer = rag_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history_str,
        }
    )

    # update memory
    memory.add_user_message(question)
    memory.add_ai_message(answer)

    return {
        "answer": answer,
        "docs": retrieved_docs,
        "chat_history": memory.messages,
    }
