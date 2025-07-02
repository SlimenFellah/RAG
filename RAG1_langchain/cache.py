import json
import os
from config import CACHE_FILE

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_cached_answer(query):
    cache = load_cache()
    return cache.get(query)

def cache_answer(query, answer):
    cache = load_cache()
    cache[query] = answer
    save_cache(cache)
