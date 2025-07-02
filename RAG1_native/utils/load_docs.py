import os

def load_markdown_docs(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".md"):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs
