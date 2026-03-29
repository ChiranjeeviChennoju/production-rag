import httpx


class OllamaClient:
    """Talks to a locally running Ollama server."""

    def __init__(self, model: str = "tinyllama", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=300
        )
        response.raise_for_status()
        return response.json()["response"]
