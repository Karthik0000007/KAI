import requests

def get_response(prompt):
    print(f"[LLM] Getting response from Ollama...")
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False
        }
    )
    reply = response.json()["response"]
    return reply.strip()
