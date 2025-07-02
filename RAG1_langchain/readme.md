# RAG1_langchain – RAG with LangChain + Ollama

This project implements a more modular and scalable **RAG pipeline using LangChain**, integrated with **local Ollama models** (`llama2`, `tinyllama`).

---

## 🔧 Tech Stack
- 📄 Document loader: `DirectoryLoader`
- ✂️ Text splitter: `RecursiveCharacterTextSplitter`
- 🧠 Embeddings: `HuggingFaceEmbeddings`
- 🔍 Vector store: `FAISS`
- 💬 LLM: `llama2` via `Ollama`
- 🧱 RAG Chain: `RetrievalQA`

---

## 🔁 Workflow
1. Loads markdown documents from `docs/`
2. Splits them into chunks for embedding
3. Converts chunks into vector embeddings
4. Stores and queries them using FAISS
5. Retrieves relevant docs using semantic similarity
6. Uses a local LLM to generate an answer with context

---
## 🚀 How to Run

1. **Clone and enter the project directory:**
    ```bash
    cd RAG1_langchain
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Start Ollama (ensure it's running):**
    ```bash
    ollama run llama2
    ```
4. **Run the main pipeline:**
    ```bash
    python main.py
    ```

---

## 📁 Project Structure

- `main.py` – Entire LangChain RAG pipeline
- `docs/` – Dummy markdown documents

---

## ⚠️ Notes on Warnings

- Some components (Ollama, HuggingFaceEmbeddings, FAISS) are deprecated and will be removed in LangChain v1.0.
- **Update imports to:**
  - `langchain_community.vectorstores`
  - `langchain_huggingface.embeddings`
  - `langchain_ollama.llms`

---

## ✅ Pros

- Cleaner abstraction
- Easy to swap LLMs, retrievers, splitters
- Supports advanced LangChain features (tracing, agents, tools)

---

## ⚠️ Limitations

- Slightly more overhead
- Deprecation warnings in current version
