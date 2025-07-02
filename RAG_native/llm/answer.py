import requests

def ask_ollama(prompt, model="llama2"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()["response"]
