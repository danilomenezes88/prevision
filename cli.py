"""
Prevision AI Engine - Interface CLI
Permite executar simulações completas de enxame preditivo diretamente do terminal.
"""

import argparse
import sys
import os

# Garante importação local correta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from llm_client import LLMClient
from graph_rag import GraphRAG
from persona_generator import PersonaGenerator
from simulation_engine import SwarmSimulationEngine
from report_agent import ReportAgent


def run_cli_simulation(scenario: str, num_agents: int = 15, num_rounds: int = 3, provider: str = None):
    print("=" * 70)
    print("🔮 PREVISION ENGINE - SIMULADOR PREDITIVO DE INTELIGÊNCIA DE ENXAME")
    print("=" * 70)

    if provider:
        config.llm_provider = provider.lower()

    print(f"\n[Configuração]")
    print(f"• Provedor de LLM : {config.llm_provider.upper()}")
    print(f"• Agentes Totais  : {num_agents}")
    print(f"• Rodadas         : {num_rounds}")
    print(f"• Cenário         : {scenario[:80]}...")

    llm = LLMClient()

    # 1. GraphRAG
    print("\n[1/4] Extraindo Grafo de Conhecimento (GraphRAG)...")
    graph_rag = GraphRAG(llm=llm)
    graph_data = graph_rag.build_from_text(scenario)
    print(f"  ✓ Grafo construído com {graph_data['num_nodes']} entidades e {graph_data['num_edges']} relações.")

    # 2. Geração de Personas
    print("\n[2/4] Instanciando População Heterogênea de Agentes...")
    generator = PersonaGenerator(llm=llm)
    agents = generator.generate_population(scenario, graph_rag, total_agents=num_agents)
    for a in agents[:5]:
        print(f"  • {a.name} | Risco: {a.risk_aversion}% | Sensibilidade a Preço: {a.price_sensitivity}% | Postura: {a.stance}")
    if len(agents) > 5:
        print(f"  ... e mais {len(agents) - 5} personas simuladas.")

    # 3. Motor de Simulação
    print("\n[3/4] Iniciando Simulação de Enxame...")
    engine = SwarmSimulationEngine(scenario=scenario, agents=agents, graph_rag=graph_rag, llm=llm)

    for r in range(1, num_rounds + 1):
        print(f"\n--- ⏳ RODADA {r}/{num_rounds} ---")
        metrics = engine.step_round(r)
        
        # Exibe as últimas postagens da rodada
        round_posts = [p for p in engine.feed if p.round_num == r]
        for p in round_posts[:4]:
            icon = "🟢" if p.sentiment > 0.2 else "🔴" if p.sentiment < -0.2 else "⚪"
            print(f"  {icon} [@{p.author_name} ({p.author_role})]: {p.content}")
        
        print(f"  >> Métrica da Rodada {r}: Sentimento Médio: {metrics.average_sentiment:+.2f} | Polarização: {metrics.polarization_index:.1f}% | Rejeição: {metrics.boycott_rate:.1f}%")

    # 4. Relatório Preditivo
    print("\n[4/4] Sintetizando Relatório Preditivo Executivo...")
    reporter = ReportAgent(simulation=engine, llm=llm)
    report = reporter.generate_prediction_report()

    print("\n" + "=" * 70)
    print("📊 RESULTADO DA PREVISÃO EXECUTIVA")
    print("=" * 70)
    print(f"• Probabilidade de Sucesso Estimada : {report.success_probability}%")
    print(f"• Nível de Risco                    : {report.risk_level.upper()}")
    print(f"• Grau de Polarização Social         : {report.polarization_degree.upper()}")
    print("-" * 70)
    print(report.raw_markdown)
    print("=" * 70)

    # Modo Interativo de Interrogação
    print("\n💡 Modo Interativo: Você pode interrogar o mundo simulado ou conversar com agentes.")
    print("Digite 'sair' para encerrar.")
    while True:
        try:
            q = input("\n[Interrogar Simulação] > ").strip()
            if not q or q.lower() in ["sair", "exit", "quit"]:
                break
            answer = reporter.interrogate_world(q)
            print(f"\n[Resposta do Oráculo Preditivo]:\n{answer}")
        except (KeyboardInterrupt, EOFError):
            break

    print("\nSimulação concluída com sucesso.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prevision Swarm Intelligence Prediction Engine CLI")
    parser.add_argument(
        "--scenario",
        type=str,
        default="A empresa de streaming StreamMax anunciou um reajuste de preço de 25% na assinatura mensal no Brasil e o bloqueio de compartilhamento de senhas.",
        help="Descrição do cenário ou decisão a ser simulada"
    )
    parser.add_argument("--agents", type=int, default=12, help="Número de personas de IA a instanciar")
    parser.add_argument("--rounds", type=int, default=3, help="Número de rodadas de simulação")
    parser.add_argument("--provider", type=str, default=None, help="Provedor LLM: openai, gemini, anthropic, groq, ollama")

    args = parser.parse_args()
    run_cli_simulation(
        scenario=args.scenario,
        num_agents=args.agents,
        num_rounds=args.rounds,
        provider=args.provider
    )
