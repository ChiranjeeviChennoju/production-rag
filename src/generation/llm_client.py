import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
from groq import Groq
from config.settings import settings


class OllamaClient:
    """Talks to a locally running Ollama server."""

    def __init__(self):
        self.model = settings.ollama_model
        self.base_url = settings.ollama_base_url

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=300
        )
        response.raise_for_status()
        return response.json()["response"]


class GroqClient:
    """
    Free cloud LLM via Groq API.
    Recommended for Intel Macs where local Ollama is too slow.
    Get a free API key at https://console.groq.com
    Set GROQ_API_KEY in your .env file.
    """

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content
