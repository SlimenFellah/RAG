from langchain_community.vectorstores import Chroma
from embedder import get_embedder
from config import CHROMA_DB_DIR

def get_vectorstore(documents):
    embedding = get_embedder()
    return Chroma.from_documents(documents, embedding, persist_directory=CHROMA_DB_DIR)

def load_vectorstore():
    embedding = get_embedder()
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding)
