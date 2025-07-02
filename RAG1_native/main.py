from utils.load_docs import load_markdown_docs
from embed.embed_docs import get_embeddings
from index.build_index import build_faiss_index
from query.ask import search_index
from sentence_transformers import SentenceTransformer
import numpy as np
from llm.answer import ask_ollama

# Load documents
docs = load_markdown_docs("../docs")

# Embed docs
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = get_embeddings(docs)

# Build FAISS index
embeddings_np = np.array(embeddings)
index = build_faiss_index(embeddings_np)

# Ask question
# query = "Where did Slimene work in 2020?"
query = "With what technology SmartRecruiter was built?"
top_docs = search_index(index, model, query, docs)

print("\nTop Relevant Documents:\n")
print("\n---\n".join(top_docs))


top_docs = search_index(index, model, query, docs)
context = "\n".join(top_docs)

# Build prompt
prompt = f"""You are an AI assistant answering based on the following documents:

{context}

Question: {query}
Answer:"""

answer = ask_ollama(prompt, model="llama2")  # or "tinyllama"
print("Answer:\n", answer)