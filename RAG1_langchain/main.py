from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1. Load markdown files from a folder
loader = DirectoryLoader('../docs', glob='**/*.md')
docs = loader.load()

# 2. Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vector Store (FAISS)
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 5. LLM via Ollama
llm = Ollama(model="llama2")  # or "tinyllama"

# 6. Retrieval-QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 7. Ask a question
query = "Where did Slimene work in 2020?"
result = qa_chain(query)

# 8. Print result
print("\nAnswer:\n", result['result'])
print("\nSources:\n", [doc.metadata['source'] for doc in result['source_documents']])
