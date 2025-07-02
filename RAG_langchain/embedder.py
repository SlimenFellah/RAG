from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
