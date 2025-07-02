def search_index(index, embedding_model, query, docs, k=2):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, k)
    return [docs[i] for i in I[0]]
