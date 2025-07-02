# import os
# from vectorstore import get_vectorstore
# from data_loader import load_and_split_documents
from rag_pipeline import ask_question
# from config import CHROMA_DB_DIR
from file_tracker import scan_documents, load_tracked_hashes, save_tracked_hashes, get_updated_files
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vectorstore import load_vectorstore
from config import DOCS_PATH
from cache_cleaner import clean_query_cache_on_doc_change

# Step 1: Build vectorstore only if it doesn't exist
# if not os.path.exists(CHROMA_DB_DIR):
#     print("Indexing documents...")
#     docs = load_and_split_documents()
#     get_vectorstore(docs).persist()
# else:
#     print("Using existing Chroma vector DB.")

current_hashes = scan_documents(DOCS_PATH)
old_hashes = load_tracked_hashes()
updated_files = get_updated_files(current_hashes, old_hashes)

if updated_files:
    print(f"\n📄 Detected {len(updated_files)} new or modified file(s):\n")
    for path in updated_files:
        print(f"  - {path}")

    proceed = input("\nProceed with indexing these files? (y/n): ").lower().strip()

    if proceed == 'y':
        vectorstore = load_vectorstore()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        all_chunks = []

        for filepath in updated_files:
            loader = TextLoader(filepath)
            docs = loader.load()

            # Tag each document with its file path
            for doc in docs:
                doc.metadata["source"] = filepath

            chunks = splitter.split_documents(docs)

            # 🧹 Remove old chunks by source
            vectorstore.delete(filter={"source": filepath})

            all_chunks.extend(chunks)

        vectorstore.add_documents(all_chunks)
        vectorstore.persist()
        clean_query_cache_on_doc_change("query_cache.json", all_chunks)
        save_tracked_hashes(current_hashes)

        print(f"\n✅ Successfully indexed {len(updated_files)} file(s).")
    else:
        print("\n⏩ Skipped indexing. No changes were made to the vector DB.")
else:
    print("✅ No new or changed files detected. Using existing Chroma vector DB.")


def get_deleted_files(old_hashes, current_hashes):
    return [path for path in old_hashes if path not in current_hashes]

deleted_files = get_deleted_files(old_hashes, current_hashes)

if deleted_files:
    print(f"\n🗑️ Detected {len(deleted_files)} deleted file(s):")
    for path in deleted_files:
        print(f"  - {path}")
    
    proceed_delete = input("\nDelete their vectors from DB? (y/n): ").lower().strip()
    if proceed_delete == 'y':
        vectorstore = load_vectorstore()
        for filepath in deleted_files:
            vectorstore.delete(filter={"source": filepath})
        print("✅ Removed vectors from deleted files.")
        # Remove cache entries possibly based on deleted files
        deleted_doc_texts = []  # Load file content if you backed it up
        # clean_query_cache_on_doc_change("query_cache.json", deleted_doc_texts)  # optional


query = input("Enter your question: ")
result = ask_question(query)

if result["cached"]:
    print("\n(Cached result)")
print("\nAnswer:\n", result["result"])