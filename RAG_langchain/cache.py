import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import CACHE_FILE


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.9 # Adjust as needed

model = SentenceTransformer(EMBEDDING_MODEL)

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_cached_answer(query):
    query_emb = model.encode([query])[0]
    cache = load_cache()

    for entry in cache:
        emb = np.array(entry["embedding"])
        sim = cosine_similarity([query_emb], [emb])[0][0]
        if sim > SIMILARITY_THRESHOLD:
            print("A similar query already exists in the cache.")
            print(f"Cached Query: {entry['query']}")
            print(f"Cached Answer: {entry['answer']}")
            return entry["answer"]

    return None

def cache_answer(query, answer):
    query_emb = model.encode([query])[0].tolist()
    cache = load_cache()
    cache.append({
        "query": query,
        "embedding": query_emb,
        "answer": answer
    })
    save_cache(cache)