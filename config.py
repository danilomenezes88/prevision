"""
MiroFish Engine - Módulo de Configuração
Gerencia chaves de API, provedores de IA e hiperparâmetros de simulação.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass


@dataclass
class Config:
    # Provedor padrão: openai | gemini | anthropic | groq | ollama
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Google Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    # Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Ollama Local
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Hiperparâmetros de Simulação
    default_num_agents: int = int(os.getenv("DEFAULT_NUM_AGENTS", "20"))
    default_num_rounds: int = int(os.getenv("DEFAULT_NUM_ROUNDS", "3"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))


# Instância global padrão
config = Config()
