# main.py

import os
from vectorstore import get_vectorstore
from data_loader import load_and_split_documents
from rag_pipeline import ask_question
from config import CHROMA_DB_DIR

# Step 1: Build vectorstore only if it doesn't exist
if not os.path.exists(CHROMA_DB_DIR):
    print("Indexing documents...")
    docs = load_and_split_documents()
    get_vectorstore(docs).persist()
else:
    print("Using existing Chroma vector DB.")

# Step 2: Ask a question
query = input("Enter your question: ")
result = ask_question(query)
print("\nAnswer:\n", result["result"])

if result["cached"]:
    print("\n(Cached result)")
print("\nAnswer:\n", result["result"])