# rag_pipeline.py

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from vectorstore import load_vectorstore
from config import OLLAMA_MODEL

def build_rag_chain():
    llm = OllamaLLM(model=OLLAMA_MODEL)
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(query: str):
    chain = build_rag_chain()
    return chain.invoke(query)
