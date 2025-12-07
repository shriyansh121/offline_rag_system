import os

# -------------------------------
# Project Structure Definition
# -------------------------------

folders = [
    "src",
    "src/config",
    "src/loaders",
    "src/splitter",
    "src/embeddings",
    "src/vectorstore",
    "src/retriever",
    "src/llm",
    "src/rag",
    "src/memory",
    "src/utils",
    "data",
    "data/pdfs",
    "data/chroma_minilm_db",
]

files = [
    "app.py",
    "requirements.txt",
    "Dockerfile",

    "src/__init__.py",
    "src/config/__init__.py",
    "src/config/settings.py",

    "src/loaders/__init__.py",
    "src/loaders/pdf_loader.py",

    "src/splitter/__init__.py",
    "src/splitter/text_splitter.py",

    "src/embeddings/__init__.py",
    "src/embeddings/minilm_embeddings.py",

    "src/vectorstore/__init__.py",
    "src/vectorstore/chroma_store.py",

    "src/retriever/__init__.py",
    "src/retriever/get_retriever.py",

    "src/llm/__init__.py",
    "src/llm/ollama_llm.py",

    "src/rag/__init__.py",
    "src/rag/rag_chain.py",

    "src/memory/__init__.py",
    "src/memory/chat_memory.py",

    "src/utils/__init__.py",
    "src/utils/helpers.py",
]

# -------------------------------
# Create folders
# -------------------------------
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

# -------------------------------
# Create empty files
# -------------------------------
for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass
        print(f"Created file: {file}")
    else:
        print(f"Skipped (already exists): {file}")

print("\n🎉 Project structure created successfully!")
