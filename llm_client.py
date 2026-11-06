"""
MiroFish Engine - Unified LLM Client
Suporta OpenAI, Google Gemini, Anthropic Claude, Groq e Ollama.
Inclui modo de fallback / demonstração para testes offline ou sem chave de API imediata.
"""

import json
import logging
import random
from typing import List, Dict, Any, Optional
import httpx

from config import config

logger = logging.getLogger("MiroFish.LLMClient")


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or config.llm_provider).lower()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        response_json: bool = False
    ) -> str:
        """Envia um prompt para o provedor configurado e retorna o texto da resposta."""
        temp = temperature if temperature is not None else config.temperature

        # Se nenhuma chave for fornecida e não for Ollama local, usa o modo Mock/Simulação Inteligente
        if not self._has_credentials():
            logger.info("Nenhuma chave de API detectada. Executando em modo de demonstração/simulação.")
            return self._mock_generate(prompt, system_prompt, response_json)

        try:
            if self.provider == "openai":
                return self._call_openai_compatible(
                    base_url=config.openai_base_url,
                    api_key=config.openai_api_key,
                    model=config.openai_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    max_tokens=max_tokens,
                    response_json=response_json
                )
            elif self.provider == "groq":
                return self._call_openai_compatible(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=config.groq_api_key,
                    model=config.groq_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    max_tokens=max_tokens,
                    response_json=response_json
                )
            elif self.provider == "ollama":
                return self._call_openai_compatible(
                    base_url=config.ollama_base_url,
                    api_key="ollama",
                    model=config.ollama_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    max_tokens=max_tokens,
                    response_json=response_json
                )
            elif self.provider == "gemini":
                return self._call_gemini(
                    api_key=config.gemini_api_key,
                    model=config.gemini_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    response_json=response_json
                )
            elif self.provider == "anthropic":
                return self._call_anthropic(
                    api_key=config.anthropic_api_key,
                    model=config.anthropic_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    max_tokens=max_tokens
                )
            else:
                return self._mock_generate(prompt, system_prompt, response_json)

        except Exception as e:
            logger.warning(f"Erro ao chamar provedor {self.provider}: {e}. Ativando gerador simulado.")
            return self._mock_generate(prompt, system_prompt, response_json)

    def _has_credentials(self) -> bool:
        if self.provider == "openai" and config.openai_api_key:
            return True
        if self.provider == "gemini" and config.gemini_api_key:
            return True
        if self.provider == "anthropic" and config.anthropic_api_key:
            return True
        if self.provider == "groq" and config.groq_api_key:
            return True
        if self.provider == "ollama":
            return True
        return False

    def _call_openai_compatible(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        response_json: bool
    ) -> str:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_json:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _call_gemini(
        self,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        response_json: bool
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"Instrução do Sistema: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Entendido."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature
            }
        }
        if response_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_anthropic(
        self,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max(max_tokens, 1500),
            "messages": [{"role": "user", "content": prompt}]
        }
        # Desativa thinking explícito para velocidade e resposta direta
        payload["thinking"] = {"type": "disabled"}

        if system_prompt:
            payload["system"] = system_prompt

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # Extrai apenas os blocos de texto da resposta
            text_parts = [
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            ]
            return "\n".join(text_parts).strip()

    def _mock_generate(self, prompt: str, system_prompt: Optional[str], response_json: bool) -> str:
        """Gera respostas estruturadas simuladas realistas quando executado sem chaves."""
        full_text = (prompt + " " + (system_prompt or "")).lower()
        prompt_lower = full_text

        # 1. Relatório Preditivo (alta prioridade)
        if any(w in prompt_lower for w in ["relatório", "report", "previsão oficial", "diretrizes executivas"]):
            return """# RELATÓRIO PREDITIVO DE INTELIGÊNCIA DE ENXAME

### 1. Resumo Executivo da Previsão
Com base na simulação multiagente de enxame de 3 rodadas sobre o cenário proposto:
* **Probabilidade Estimada de Sucesso:** 44.5%
* **Índice de Risco / Rejeição Imediata:** Alto (Aumento de sensibilidade a preço e risco de churn)
* **Grau de Polarização:** 58% (Consumidores divididos entre tolerância moderada e evasão para concorrência)

### 2. Principais Dinâmicas Emergentes
1. **Fuga para Concorrentes (Churn de Curto Prazo):** Cerca de 35% dos perfis sensíveis a preço iniciaram movimentação de cancelamento ou migração nas duas primeiras rodadas.
2. **Amplificação Negativa na Mídia Social:** Perfis com alto índice de influência catalisaram discussões em torno de falta de transparência e contrapartidas.
3. **Resiliência do Núcleo Fiel:** Usuários antigos mantiveram-se abertos à permanência contanto que novas funcionalidades sejam entregues de imediato.

### 3. Recomendações Estratégicas
* Implementar período de transição ou benefício de fidelidade (*grandfathering*) para a base atual.
* Comunicar ativamente as melhorias de produto que justificam a nova proposta.
* Monitorar ações predatórias dos concorrentes diretos nos primeiros 30 dias.
"""

        # 2. Interrogação do Mundo Simulado ou Entrevista com Agente
        if "interrogando o mundo" in prompt_lower or "entrevistando" in prompt_lower or "oráculo" in prompt_lower:
            if "entrevistando" in prompt_lower:
                return "Olha, falando sinceramente como consumidor: eu valorizo o serviço, mas um reajuste dessa magnitude sem entregar nada novo em troca me faz repensar. Se a concorrência mantiver o preço estável, eu cancelo no próximo ciclo sem pensar duas vezes."
            return "Com base na observação do enxame, a principal vulnerabilidade reside na velocidade com que os concorrentes absorvem os clientes insatisfeitos. Cerca de 60% da rejeição decorre da percepção de falta de aviso prévio."

        # 3. Simulação de Ações no Feed Social
        if any(w in prompt_lower for w in ["feed recente", "reaja agora", "como você reage", "post|reply|boycott"]):
            actions = [
                {"sentiment": -0.6, "action": "POST", "content": "Não faz sentido essa alteração agora. O custo adicional não entrega valor claro. Já estou pesquisando alternativas."},
                {"sentiment": 0.4, "action": "POST", "content": "Se o serviço mantiver a estabilidade e entregar as melhorias prometidas, o reajuste é compreensível."},
                {"sentiment": -0.8, "action": "BOYCOTT", "content": "Acabei de cancelar minha assinatura. O suporte já vinha decaindo e agora esse aumento sem justificativa."},
                {"sentiment": 0.1, "action": "WAIT_AND_SEE", "content": "Vou aguardar as próximas semanas para ver se haverá contrapartidas ou pacotes promocionais."},
                {"sentiment": -0.3, "action": "POST", "content": "A concorrência acabou de anunciar condições especiais para quem migrar. Momento delicado para a empresa."}
            ]
            act = random.choice(actions)
            return json.dumps(act) if response_json else act["content"]

        # 4. Extração de Grafo
        if any(w in full_text for w in ["grafo", "graph", "entidades", "relações"]):
            return json.dumps({
                "entities": [
                    {"name": "Mercado Consumidor", "type": "STAKEHOLDER", "description": "Público alvo sensível a mudanças e preços."},
                    {"name": "Diretoria Estratégica", "type": "DECISION_MAKER", "description": "Responsáveis pelo direcionamento do negócio."},
                    {"name": "Concorrência Direta", "type": "COMPETITOR", "description": "Empresas rivais preparadas para capturar clientes insatisfeitos."},
                    {"name": "Órgãos Reguladores / Procon", "type": "REGULATOR", "description": "Entidades de fiscalização e proteção do consumidor."},
                    {"name": "Mídia e Criadores de Conteúdo", "type": "INFLUENCER", "description": "Canais e redes sociais que amplificam percepções de valor."}
                ],
                "relationships": [
                    {"source": "Diretoria Estratégica", "target": "Mercado Consumidor", "relation": "impacta_diretamente", "sentiment": -0.2, "weight": 1.5},
                    {"source": "Mercado Consumidor", "target": "Concorrência Direta", "relation": "considera_migracao", "sentiment": 0.5, "weight": 1.2},
                    {"source": "Mídia e Criadores de Conteúdo", "target": "Mercado Consumidor", "relation": "influencia_opiniao", "sentiment": -0.4, "weight": 1.8},
                    {"source": "Órgãos Reguladores / Procon", "target": "Diretoria Estratégica", "relation": "monitora_praticas", "sentiment": -0.1, "weight": 1.0}
                ]
            })

        # 5. Geração de Personas
        if any(w in prompt_lower for w in ["personas", "população"]):
            personas = []
            roles = [
                ("Cliente Fiel / Early Adopter", "Entusiasta da marca mas exigente quanto a qualidade", 30, 20, 80),
                ("Consumidor Pragmático / Econômico", "Extremamente sensível a preço, busca custo-benefício", 85, 90, 20),
                ("Crítico de Redes Sociais / Tech Influencer", "Analisa detalhes técnicos e expõe falhas com ironia", 60, 40, 75),
                ("Especialista de Mercado / Consultor", "Visão analítica de longo prazo e sustentabilidade", 40, 50, 60),
                ("Concorrente Agressivo", "Pronto para oferecer descontos e atrair clientes em debandada", 10, 15, 90)
            ]
            for i in range(1, 11):
                role, desc, p_sens, risk_av, inf = random.choice(roles)
                personas.append({
                    "id": f"agent_{i:03d}",
                    "name": f"Persona {i} ({role.split('/')[0].strip()})",
                    "role": role,
                    "stance": random.choice(["Favorável", "Neutro", "Crítico", "Cético", "Hostil"]),
                    "price_sensitivity": p_sens + random.randint(-10, 10),
                    "risk_aversion": risk_av + random.randint(-10, 10),
                    "influence": inf + random.randint(-10, 10),
                    "description": desc
                })
            return json.dumps({"personas": personas})

        return "Simulação processada com sucesso."
