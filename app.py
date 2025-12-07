import streamlit as st

from src.config.settings import APP_NAME, PDF_DIR
from src.loaders.pdf_loader import load_pdfs
from src.splitter.text_splitter import split_docs
from src.vectorstore.chroma_store import load_or_create_vectorstore
from src.retriever.get_retriever import get_similarity_retriever, get_mmr_retriever
from src.llm.ollama_llm import get_llm
from src.memory.chat_memory import get_memory
from src.rag.rag_chain import run_rag_with_memory
from src.utils.helpers import extract_sources


# -------------------------
# Caching heavy components
# -------------------------
@st.cache_resource(show_spinner=True)
def get_docs_and_vectorstore():
    docs = load_pdfs(PDF_DIR)
    splits = split_docs(docs)
    vectorstore = load_or_create_vectorstore(splits)
    return docs, splits, vectorstore


@st.cache_resource(show_spinner=True)
def get_retrievers():
    _, _, vectorstore = get_docs_and_vectorstore()
    retriever_sim = get_similarity_retriever(vectorstore)
    retriever_mmr = get_mmr_retriever(vectorstore)
    return retriever_sim, retriever_mmr


@st.cache_resource(show_spinner=True)
def get_llm_cached():
    return get_llm()


def init_session_state():
    if "memory" not in st.session_state:
        st.session_state.memory = get_memory()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {"role": "user"/"assistant", "content": ...}
    if "debug_show_context" not in st.session_state:
        st.session_state.debug_show_context = False
    if "retriever_type" not in st.session_state:
        st.session_state.retriever_type = "MMR"


def main():
    st.set_page_config(page_title=APP_NAME, layout="wide")
    init_session_state()

    st.title(APP_NAME)
    st.caption("Offline RAG using PDFs + Chroma + MiniLM + Gemma3:4b (Ollama)")

    # Sidebar
    with st.sidebar:
        st.subheader("Settings")

        st.write("**PDF directory:**")
        st.code(PDF_DIR, language="bash")

        retriever_type = st.radio(
            "Retriever type",
            options=["MMR (diverse)", "Similarity"],
            index=0 if st.session_state.retriever_type == "MMR" else 1,
        )
        st.session_state.retriever_type = "MMR" if "MMR" in retriever_type else "Similarity"

        st.session_state.debug_show_context = st.checkbox(
            "Show retrieved chunks (debug)", value=st.session_state.debug_show_context
        )

        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.session_state.memory = get_memory()
            st.success("Chat history cleared!")

        st.markdown("---")
        st.markdown("**Status**")
        with st.spinner("Loading docs & vectorstore..."):
            docs, splits, vectorstore = get_docs_and_vectorstore()
        st.write(f"Documents loaded: **{len(docs)}**")
        st.write(f"Chunks created: **{len(splits)}**")

    # Main retrievers / llm
    retriever_sim, retriever_mmr = get_retrievers()
    llm = get_llm_cached()
    memory = st.session_state.memory

    retriever = retriever_mmr if st.session_state.retriever_type == "MMR" else retriever_sim

    # Chat display area
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask something about your PDFs...")
    if user_input:
        # display user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # RAG call
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_rag_with_memory(
                    question=user_input,
                    retriever=retriever,
                    llm=llm,
                    memory=memory,
                )
                answer = result["answer"]
                docs = result["docs"]

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Sources
                sources = extract_sources(docs)
                if sources:
                    with st.expander("Sources (PDFs & pages)"):
                        for src in sources:
                            line = f"- **{src['source']}**"
                            if src["page"] is not None:
                                line += f", page {src['page']}"
                            st.markdown(line)
                            st.caption(src["snippet"])

                # Debug: raw context
                if st.session_state.debug_show_context:
                    with st.expander("Debug: raw retrieved text chunks"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}** — {src['source']} (page {src['page']})")
                            st.write(src["snippet"])


if __name__ == "__main__":
    main()
