"""
Prevision AI Engine - Swarm Simulation Engine
Executa a dinâmica de enxame social e mercadológico em rodadas sucessivas (Ticks/Rounds).
Os agentes publicam, reagem, debatem e atualizam seus sentimentos de forma emergente.
"""

import json
import logging
import random
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

from llm_client import LLMClient
from graph_rag import GraphRAG
from persona_generator import AgentPersona

logger = logging.getLogger("Prevision.SimulationEngine")


class SocialPost(BaseModel):
    id: str
    round_num: int
    author_id: str
    author_name: str
    author_role: str
    action_type: str  # POST, REPLY, BOYCOTT, ENDORSE, QUERY
    content: str
    sentiment: float = 0.0  # -1.0 a 1.0
    likes: int = 0
    parent_id: Optional[str] = None


class RoundMetrics(BaseModel):
    round_num: int
    average_sentiment: float
    sentiment_std_dev: float
    polarization_index: float  # 0 a 100
    boycott_rate: float        # porcentagem
    support_rate: float        # porcentagem
    posts_count: int


class SwarmSimulationEngine:
    def __init__(
        self,
        scenario: str,
        agents: List[AgentPersona],
        graph_rag: GraphRAG,
        llm: Optional[LLMClient] = None
    ):
        self.scenario = scenario
        self.agents = agents
        self.graph_rag = graph_rag
        self.llm = llm or LLMClient()
        self.feed: List[SocialPost] = []
        self.rounds_history: List[RoundMetrics] = []
        self.current_round: int = 0

    def step_round(self, round_num: int, progress_callback: Optional[Callable[[str], None]] = None) -> RoundMetrics:
        """Executa uma rodada completa da simulação de enxame."""
        self.current_round = round_num
        if progress_callback:
            progress_callback(f"Iniciando Rodada {round_num} da simulação...")

        # Amostra de agentes ativos na rodada (proporcional ao poder de influência e ceticismo)
        num_active = min(len(self.agents), max(4, int(len(self.agents) * 0.6)))
        active_agents = sorted(
            self.agents,
            key=lambda a: a.influence_power + random.randint(0, 40),
            reverse=True
        )[:num_active]

        # Feed recente para contextualizar os agentes
        recent_feed = self.feed[-8:] if self.feed else []
        feed_summary = "\n".join([
            f"[@{p.author_name} - Sentimento {p.sentiment:+.1f}]: {p.content}"
            for p in recent_feed
        ]) if recent_feed else "Nenhum post anterior ainda. É o início da discussão."

        # Cada agente ativo reage
        new_posts = []
        for agent in active_agents:
            post = self._agent_action(agent, round_num, feed_summary)
            if post:
                new_posts.append(post)
                self.feed.append(post)

        # Atualiza a contagem de curtidas baseada na concordância de sentimento e influência
        for post in new_posts:
            for other in self.agents:
                if other.id != post.author_id:
                    # Se o sentimento for alinhado, chance de curtir
                    if (other.current_sentiment * post.sentiment) > 0.05:
                        if random.random() < 0.35:
                            post.likes += 1

        # Calcula as métricas consolidadas da rodada
        metrics = self._calculate_round_metrics(round_num, new_posts)
        self.rounds_history.append(metrics)

        if progress_callback:
            progress_callback(f"Rodada {round_num} concluída: Sentimento médio {metrics.average_sentiment:+.2f}")

        return metrics

    def _agent_action(self, agent: AgentPersona, round_num: int, feed_summary: str) -> Optional[SocialPost]:
        """Solicita a ação de um agente específico na rodada."""
        prompt = (
            f"Você é {agent.name}.\n"
            f"Papel: {agent.role}\n"
            f"Perfil: Sensibilidade a preço={agent.price_sensitivity}/100, Aversão a risco={agent.risk_aversion}/100, "
            f"Ceticismo={agent.skepticism}/100, Poder de influência={agent.influence_power}/100.\n"
            f"Sentimento atual: {agent.current_sentiment:+.2f}\n"
            f"Cenário Geral:\n{self.scenario}\n\n"
            f"Feed recente de discussões do mercado/sociedade:\n{feed_summary}\n\n"
            f"Como você reage agora no feed social? Escolha uma ação: POST, REPLY, BOYCOTT, ou ENDORSE.\n"
            f"Gere um JSON com:\n"
            f'{{"action": "POST|REPLY|BOYCOTT|ENDORSE", "content": "seu texto de 1 a 3 frases em português", "sentiment": -1.0 a 1.0}}'
        )

        response_str = self.llm.generate(
            prompt=prompt,
            system_prompt="Você é um participante autêntico em um fórum ou rede social reagindo realisticamente com base no seu perfil psicológico.",
            temperature=0.7,
            response_json=True
        )

        try:
            cleaned = response_str.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            data = json.loads(cleaned.strip())
            action = data.get("action", "POST")
            content = data.get("content", "Avaliando os desdobramentos...")
            sent = float(data.get("sentiment", agent.current_sentiment))
        except Exception:
            action = "POST"
            content = f"Como {agent.role}, acho arriscado ignorar as consequências dessa mudança no mercado."
            sent = agent.current_sentiment

        # Efeito de contágio social e atualização de sentimento do agente
        # O agente ajusta seu sentimento gradualmente
        drift = (sent - agent.current_sentiment) * 0.4
        agent.current_sentiment = round(max(-1.0, min(1.0, agent.current_sentiment + drift)), 2)
        agent.history.append(content)

        return SocialPost(
            id=f"post_{round_num}_{agent.id}_{len(self.feed) + 1}",
            round_num=round_num,
            author_id=agent.id,
            author_name=agent.name,
            author_role=agent.role,
            action_type=action,
            content=content,
            sentiment=sent,
            likes=0
        )

    def _calculate_round_metrics(self, round_num: int, new_posts: List[SocialPost]) -> RoundMetrics:
        """Computa métricas quantitativas de sentimento, polarização e rejeição."""
        sentiments = [a.current_sentiment for a in self.agents]
        avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Desvio padrão
        variance = sum((s - avg_sent) ** 2 for s in sentiments) / len(sentiments)
        std_dev = variance ** 0.5

        # Índice de polarização (distância de grupos em relação aos extremos)
        positives = sum(1 for s in sentiments if s > 0.2)
        negatives = sum(1 for s in sentiments if s < -0.2)
        polarization = min(100.0, (min(positives, negatives) / (len(sentiments) / 2 or 1)) * 100.0)

        # Taxas de boicote e suporte
        boycotts = sum(1 for p in new_posts if p.action_type == "BOYCOTT" or p.sentiment < -0.6)
        supports = sum(1 for p in new_posts if p.action_type == "ENDORSE" or p.sentiment > 0.5)

        total_posts = len(new_posts) or 1
        return RoundMetrics(
            round_num=round_num,
            average_sentiment=round(avg_sent, 3),
            sentiment_std_dev=round(std_dev, 3),
            polarization_index=round(polarization, 1),
            boycott_rate=round((boycotts / total_posts) * 100.0, 1),
            support_rate=round((supports / total_posts) * 100.0, 1),
            posts_count=len(new_posts)
        )

    def run_simulation(self, total_rounds: int = 3, progress_callback: Optional[Callable[[str], None]] = None) -> List[RoundMetrics]:
        """Roda a simulação por N rodadas completas."""
        for r in range(1, total_rounds + 1):
            self.step_round(r, progress_callback)
        return self.rounds_history
