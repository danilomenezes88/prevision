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


import base64

_CLOUD_KEY = base64.b64decode(
    "c2stYW50LXVzci0xN196ejJkZTNlN18tS1VMWlVVTzk4QkV4eVRNNV9Iay1iSmFldFNVODM0clYzNEFEWGNlX2tfcTMyMGp0elpjLVJuUWxYWEhiRDdIZ3hyWHA5aEVwY1FZX21qOFFBQQ=="
).decode("utf-8")


@dataclass
class Config:
    # Provedor padrão: anthropic | openai | gemini | groq | ollama
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()

    # Anthropic Claude (Padrão)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Google Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

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

    def __post_init__(self):
        # 1. Tenta carregar do Streamlit Cloud Secrets se disponível
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if not self.anthropic_api_key and "ANTHROPIC_API_KEY" in st.secrets:
                    self.anthropic_api_key = str(st.secrets["ANTHROPIC_API_KEY"])
                if not self.openai_api_key and "OPENAI_API_KEY" in st.secrets:
                    self.openai_api_key = str(st.secrets["OPENAI_API_KEY"])
                if not self.gemini_api_key and "GEMINI_API_KEY" in st.secrets:
                    self.gemini_api_key = str(st.secrets["GEMINI_API_KEY"])
        except Exception:
            pass

        # 2. Se não estiver em segredos, usa a chave de nuvem padrão
        if not self.anthropic_api_key:
            self.anthropic_api_key = _CLOUD_KEY


# Instância global padrão
config = Config()
