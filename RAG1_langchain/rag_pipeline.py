from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from vectorstore import load_vectorstore
from config import OLLAMA_MODEL
from cache import get_cached_answer, cache_answer


def build_rag_chain():
    llm = OllamaLLM(model=OLLAMA_MODEL)
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(query: str):
    # Check cache
    cached = get_cached_answer(query)
    if cached:
        return {"result": cached, "cached": True}

    # Run chain
    chain = build_rag_chain()
    result = chain.invoke(query)["result"]

    # Save to cache
    cache_answer(query, result)
    return {"result": result, "cached": False}