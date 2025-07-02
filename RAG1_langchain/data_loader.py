from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import DOCS_PATH

def load_and_split_documents():
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = doc.metadata.get("source", doc.metadata.get("path", "unknown"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

