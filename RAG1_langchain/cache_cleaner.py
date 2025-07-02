import json
import os
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")  # same as embedder

def clean_query_cache_on_doc_change(cache_path, modified_docs, similarity_threshold=0.8):
    if not os.path.exists(cache_path):
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    cleaned_cache = []
    removed = 0

    for item in cache:
        answer = item.get("answer", "")
        answer_embedding = model.encode(answer, convert_to_tensor=True)

        # Check against all new/modified documents
        for doc in modified_docs:
            content = doc.page_content
            content_embedding = model.encode(content, convert_to_tensor=True)

            similarity = util.cos_sim(answer_embedding, content_embedding).item()
            if similarity > similarity_threshold:
                print(f"🧹 Removed cached answer: \"{answer}\" (similarity {similarity:.2f})")
                removed += 1
                break
        else:
            cleaned_cache.append(item)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_cache, f, indent=2)

    print(f"🧼 Cache cleanup complete: {removed} outdated entry(ies) removed.")
