from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from vectorstore import load_vectorstore
from config import OLLAMA_MODEL
from cache import get_cached_answer, cache_answer
from langchain.chains.question_answering import load_qa_chain

def build_rag_chain():
    llm = OllamaLLM(model=OLLAMA_MODEL)
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(query: str, override_docs=None):
    # Try semantic cache only if not override
    if override_docs is None:
        cached = get_cached_answer(query)
        if cached:
            return {"result": cached, "cached": True}

    # Build model and retriever
    llm = OllamaLLM(model=OLLAMA_MODEL)

    if override_docs is not None:
        # RAG manually over provided documents
        chain = load_qa_chain(llm, chain_type="stuff")
        result = chain.run(input_documents=override_docs, question=query)
    else:
        # Default RAG using vectorstore
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        print("\n🔎 Top retrieved documents:\n")
        for i, d in enumerate(docs):
            print(f"--- Doc {i+1} ---\n{d.page_content}\n")
        print("\n")
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = chain.invoke(query)["result"]

        # Save to cache only if full pipeline used
        cache_answer(query, result)

    return {"result": result, "cached": False}
