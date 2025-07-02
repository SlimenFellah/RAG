import json
import os
from typing import List
from langchain_core.documents import Document
from rag_pipeline import ask_question  # Adjust import to your actual method

def clean_query_cache_on_doc_change(cache_path: str, modified_docs: List[Document], threshold: float = 0.9):
    if not os.path.exists(cache_path):
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # Combine updated doc content into a mini knowledge base
    updated_context = "\n".join([doc.page_content for doc in modified_docs])

    cleaned_cache = []
    removed = 0

    for item in cache:
        query = item.get("query", "")
        cached_answer = item.get("answer", "").strip()

        # Re-answer the query using the updated context only
        new_answer_result = ask_question(query, override_docs=modified_docs)
        new_answer = new_answer_result["result"].strip()

        if new_answer.lower() != cached_answer.lower():
            print(f"🧹 Removed cache: '{query}'")
            print(f"   ⛔ Old: {cached_answer}")
            print(f"   ✅ New: {new_answer}")
            removed += 1
        else:
            cleaned_cache.append(item)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_cache, f, indent=2)

    print(f"🧼 Cache cleanup complete: {removed} outdated entry(ies) removed.")
