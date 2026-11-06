"""
Prevision AI Engine - Persona Generator
Gera populações heterogêneas de agentes autônomos com personalidades psicológicas,
vieses, convicções e níveis de influência a partir do cenário e do Grafo de Conhecimento.
"""

import json
import random
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from llm_client import LLMClient
from graph_rag import GraphRAG

logger = logging.getLogger("Prevision.PersonaGenerator")


class AgentPersona(BaseModel):
    id: str
    name: str
    role: str
    age: int = 35
    income_level: str = "Médio"
    region: str = "Brasil - Nacional"
    risk_aversion: int = Field(default=50, ge=0, le=100)
    price_sensitivity: int = Field(default=50, ge=0, le=100)
    influence_power: int = Field(default=50, ge=0, le=100)
    skepticism: int = Field(default=50, ge=0, le=100)
    stance: str = "Neutro"  # Favorável, Neutro, Cético, Opositor, Hostil
    current_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    bio: str = ""
    history: List[str] = Field(default_factory=list)


class PersonaGenerator:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def generate_population(
        self,
        scenario_description: str,
        graph_rag: GraphRAG,
        total_agents: int = 20
    ) -> List[AgentPersona]:
        """
        Cria uma população balanceada e realista de agentes com base no cenário e relações do grafo.
        """
        archetypes = [
            ("Consumidor Fiel / Early Adopter", "Entusiasta que valoriza inovação e qualidade acima de preço.", 25, 20, 65, 30, "Favorável", 0.4),
            ("Consumidor Pragmático / Sensível a Preço", "Foco total no custo-benefício. Qualquer aumento sem valor percebido gera rejeição.", 60, 90, 30, 70, "Cético", -0.2),
            ("Crítico de Redes Sociais / Influenciador", "Voz amplificada na internet, busca inconsistências e vocaliza a revolta da base.", 40, 50, 85, 80, "Opositor", -0.5),
            ("Analista de Mercado / Especialista", "Avalia sustentabilidade, margens e estratégia a longo prazo.", 30, 40, 70, 50, "Neutro", 0.0),
            ("Estrategista de Concorrente Direto", "Monitora a vulnerabilidade do competidor para atrair clientes insatisfeitos.", 15, 10, 75, 40, "Hostil", -0.6),
            ("Cidadão Comum / Usuário Moderado", "Usa o serviço de forma casual, reage de forma passiva mas pode migrar se incentivado.", 50, 60, 20, 40, "Neutro", 0.0),
            ("Auditor / Regulador / Defesa do Consumidor", "Preocupado com conformidade, termos de adesão e práticas abusivas.", 85, 30, 60, 90, "Cético", -0.3),
        ]

        # Gera personas variando os arquétipos e atributos
        agents: List[AgentPersona] = []
        for i in range(1, total_agents + 1):
            arch = archetypes[(i - 1) % len(archetypes)]
            role_name, base_bio, risk, price, inf, skep, initial_stance, init_sent = arch

            # Variação estocástica para evitar cópias idênticas
            agent_risk = max(5, min(95, risk + random.randint(-12, 12)))
            agent_price = max(5, min(95, price + random.randint(-12, 12)))
            agent_inf = max(5, min(95, inf + random.randint(-12, 12)))
            agent_skep = max(5, min(95, skep + random.randint(-12, 12)))
            agent_sent = max(-1.0, min(1.0, init_sent + random.uniform(-0.15, 0.15)))

            first_names = ["Carlos", "Mariana", "Rodrigo", "Camila", "Lucas", "Fernanda", "Eduardo", "Beatriz", "Gabriel", "Juliana", "Renato", "Patrícia"]
            last_names = ["Silva", "Santos", "Oliveira", "Souza", "Pereira", "Lima", "Ferreira", "Costa", "Rodrigues", "Almeida"]
            full_name = f"{random.choice(first_names)} {random.choice(last_names)}"

            agents.append(
                AgentPersona(
                    id=f"agent_{i:03d}",
                    name=f"{full_name} ({role_name.split('/')[0].strip()})",
                    role=role_name,
                    age=random.randint(22, 65),
                    income_level=random.choice(["Baixa", "Média-Baixa", "Média", "Média-Alta", "Alta"]),
                    risk_aversion=agent_risk,
                    price_sensitivity=agent_price,
                    influence_power=agent_inf,
                    skepticism=agent_skep,
                    stance=initial_stance,
                    current_sentiment=round(agent_sent, 2),
                    bio=f"{base_bio} Localização: {random.choice(['São Paulo/SP', 'Rio de Janeiro/RJ', 'Belo Horizonte/MG', 'Curitiba/PR', 'Salvador/BA'])}."
                )
            )

        logger.info(f"População de {len(agents)} agentes instanciada com sucesso.")
        return agents
