from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "tinyllama"
    groq_model: str = "llama-3.1-8b-instant"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
