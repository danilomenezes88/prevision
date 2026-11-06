"""
Prevision AI Engine - Report & Prediction Agent
Analisa o comportamento emergente do enxame, calcula probabilidades preditivas,
compila relatórios executivos e permite interrogar o mundo simulado e seus agentes.
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from llm_client import LLMClient
from simulation_engine import SwarmSimulationEngine, RoundMetrics
from persona_generator import AgentPersona

logger = logging.getLogger("Prevision.ReportAgent")


class PredictionReport(BaseModel):
    executive_summary: str
    success_probability: float  # 0 a 100%
    risk_level: str             # Baixo, Moderado, Alto, Crítico
    polarization_degree: str    # Baixa, Média, Alta
    emergent_insights: List[str]
    strategic_recommendations: List[str]
    raw_markdown: str


class ReportAgent:
    def __init__(self, simulation: SwarmSimulationEngine, llm: Optional[LLMClient] = None):
        self.sim = simulation
        self.llm = llm or simulation.llm

    def generate_prediction_report(self) -> PredictionReport:
        """Gera o relatório executivo preditivo completo baseado em todas as rodadas."""
        history = self.sim.rounds_history
        feed = self.sim.feed
        scenario = self.sim.scenario

        final_sentiment = history[-1].average_sentiment if history else 0.0
        final_polarization = history[-1].polarization_index if history else 50.0
        final_boycott = history[-1].boycott_rate if history else 0.0

        # Cálculo heurístico de probabilidade de sucesso
        # Sentimento varia de -1.0 a +1.0. Normalizamos para 0 a 100 com penalidade por boicote
        base_score = ((final_sentiment + 1.0) / 2.0) * 100.0
        success_prob = max(5.0, min(95.0, base_score - (final_boycott * 0.4)))

        if success_prob >= 70:
            risk_level = "Baixo"
        elif success_prob >= 45:
            risk_level = "Moderado"
        elif success_prob >= 25:
            risk_level = "Alto"
        else:
            risk_level = "Crítico"

        polarization_label = "Alta" if final_polarization > 60 else "Média" if final_polarization > 30 else "Baixa"

        # Amostra das mensagens mais curtidas e mais polêmicas
        top_posts = sorted(feed, key=lambda p: p.likes, reverse=True)[:6]
        top_posts_str = "\n".join([
            f"- @{p.author_name} ({p.author_role}, {p.likes} curtidas, sent={p.sentiment:+.1f}): \"{p.content}\""
            for p in top_posts
        ])

        system_prompt = (
            "Você é o Chief Predictive Officer e Analista Chefe do motor Prevision de Inteligência de Enxame. "
            "Sua especialidade é analisar comportamentos emergentes em simulações sociais/mercadológicas "
            "e gerar relatórios preditivos rigorosos, pragmáticos e de alto impacto executivo para tomada de decisão."
        )

        user_prompt = f"""
Analise os resultados da simulação de enxame multiagente e redija o Relatório Preditivo Oficial.

CENÁRIO ANALISADO:
{scenario}

DADOS DA SIMULAÇÃO:
- Total de Agentes: {len(self.sim.agents)}
- Rodadas Executadas: {len(history)}
- Sentimento Médio Final: {final_sentiment:+.2f} (Escala de -1.0 a +1.0)
- Grau de Polarização: {final_polarization:.1f}%
- Taxa Média de Boicote/Rejeição: {final_boycott:.1f}%
- Probabilidade Estimada de Adoção/Sucesso: {success_prob:.1f}%
- Nível de Risco: {risk_level}

POSTAGENS MAIS INFLUENTES NO FEED:
{top_posts_str}

ESTRUTURE SEU RELATÓRIO EM MARKDOWN COM OS SEGUINTES TÓPICOS:
1. Resumo Executivo da Previsão (com as probabilidades e diagnóstico central)
2. Dinâmicas Emergentes e Fenômenos de Enxame (como o consenso ou conflito se formou)
3. Principais Gatilhos de Risco e Pontos Cegos Detectados
4. Recomendações Estratégicas e Plano de Ação Mitigatório
"""

        raw_report = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=2500
        )

        return PredictionReport(
            executive_summary=f"Previsão de sucesso estimada em {success_prob:.1f}% com nível de risco {risk_level}.",
            success_probability=round(success_prob, 1),
            risk_level=risk_level,
            polarization_degree=polarization_label,
            emergent_insights=[
                f"Sentimento convergiu para {final_sentiment:+.2f}",
                f"Grau de polarização social atingiu {final_polarization:.1f}%",
                f"Rejeição explícita detectada em {final_boycott:.1f}% dos discursos"
            ],
            strategic_recommendations=[
                "Alinhar comunicação antecipada com os grupos mais céticos.",
                "Oferecer contrapartidas claras antes da implementação do cenário."
            ],
            raw_markdown=raw_report
        )

    def interrogate_world(self, question: str) -> str:
        """Permite ao usuário fazer perguntas analíticas sobre o mundo simulado."""
        feed_summary = "\n".join([f"[@{p.author_name}]: {p.content}" for p in self.sim.feed[-15:]])

        prompt = f"""
O usuário está interrogando o mundo simulado do Prevision com a seguinte pergunta:
"{question}"

Cenário Base:
{self.sim.scenario}

Comportamento recente dos agentes:
{feed_summary}

Responda com base nas evidências empíricas observadas durante as rodadas da simulação.
"""
        return self.llm.generate(
            prompt=prompt,
            system_prompt="Você é o oráculo analista que observou todas as interações do enxame.",
            temperature=0.5
        )

    def chat_with_agent(self, agent_id: str, message: str) -> str:
        """Permite conversar diretamente e entrevistar um agente específico da simulação."""
        agent = next((a for a in self.sim.agents if a.id == agent_id), None)
        if not agent:
            return f"Agente com ID '{agent_id}' não encontrado."

        prompt = f"""
Você é {agent.name}.
Papel: {agent.role}
Biografia: {agent.bio}
Seu histórico de comentários na simulação:
{agent.history}
Seu sentimento atual: {agent.current_sentiment:+.2f}

O usuário está te entrevistando e perguntou:
"{message}"

Responda em primeira pessoa, mantendo total fidelidade à sua personalidade, convicções e sentimentos.
"""
        return self.llm.generate(
            prompt=prompt,
            system_prompt=f"Você é {agent.name}, um participante na simulação.",
            temperature=0.7
        )
