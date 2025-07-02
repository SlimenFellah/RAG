# RAG1 – Native Python RAG Pipeline (No LangChain)

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using **native Python**, **FAISS**, and **local LLMs** via **Ollama**.

---

## 🔧 Tech Stack
- 🧾 Markdown file loader (`os`, `glob`)
- ✂️ Text splitting (`nltk`)
- 🔍 Vector search (`FAISS`)
- 🧠 Embeddings: `all-MiniLM-L6-v2` via `sentence-transformers`
- 💬 LLM: `llama2` (local, via Ollama API)

---

## 🔁 Workflow
1. Loads markdown files (e.g. `bio.md`) from the `docs/` folder.
2. Splits text into overlapping chunks.
3. Converts chunks into vector embeddings.
4. Stores and queries embeddings with FAISS.
5. Sends top results + query to a local LLM (`llama2` via Ollama).
6. Returns a generated answer based on the documents.

---

## 🚀 How to Run

1. **Install dependencies**
    ```bash
    cd RAG1
    pip install -r requirements.txt
    ```

2. **Start the Ollama server**  
   Make sure Ollama is running and the `llama2` model is available:
    ```bash
    ollama run llama2
    ```

3. **Run the pipeline**
    ```bash
    python main.py
    ```

---

## 📁 Project Structure

- `docs/` — Markdown files with dummy data
- `embed.py` — Embedding and FAISS logic
- `answer.py` — LLM generation via Ollama API
- `main.py` — Main RAG pipeline

---

## ✅ Pros

- Minimal dependencies
- Transparent and flexible
- Good for learning the mechanics of RAG

---

## ⚠️ Limitations

- Manual pipeline management
- No abstraction layers
- No automatic document loading

