import os
import hashlib
import json

TRACK_FILE = ".file_hashes.json"

def compute_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def scan_documents(directory):
    file_hashes = {}
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith(".md"):
                full_path = os.path.join(root, name)
                file_hashes[full_path] = compute_hash(full_path)
    return file_hashes

def load_tracked_hashes():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r") as f:
            return json.load(f)
    return {}

def save_tracked_hashes(hashes):
    with open(TRACK_FILE, "w") as f:
        json.dump(hashes, f, indent=2)

def get_updated_files(current_hashes, old_hashes):
    return [
        path for path, h in current_hashes.items()
        if path not in old_hashes or old_hashes[path] != h
    ]