from sentence_transformers import SentenceTransformer

def get_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings
