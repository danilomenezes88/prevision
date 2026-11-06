"""
Testes automatizados para o Prevision AI Engine.
Valida o pipeline completo: GraphRAG -> Personas -> Simulação -> Relatório -> Biblioteca de Dados.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from llm_client import LLMClient
from graph_rag import GraphRAG
from persona_generator import PersonaGenerator
from simulation_engine import SwarmSimulationEngine
from report_agent import ReportAgent
from document_store import DocumentStore


class TestPrevisionEngine(unittest.TestCase):
    def setUp(self):
        self.llm = LLMClient()
        self.scenario = (
            "A operadora de telecomunicações FibraBrasil anunciou aumento de 20% "
            "no valor dos planos de internet residencial e franquia de consumo de dados."
        )

    def test_graph_rag_extraction(self):
        graph_rag = GraphRAG(llm=self.llm)
        data = graph_rag.build_from_text(self.scenario)
        self.assertGreaterEqual(data["num_nodes"], 2, "Deveria extrair ao menos 2 entidades")
        self.assertGreaterEqual(data["num_edges"], 1, "Deveria extrair ao menos 1 relação")

    def test_persona_generation(self):
        graph_rag = GraphRAG(llm=self.llm)
        graph_rag.build_from_text(self.scenario)
        gen = PersonaGenerator(llm=self.llm)
        agents = gen.generate_population(self.scenario, graph_rag, total_agents=8)
        self.assertEqual(len(agents), 8, "Deveria instanciar exatamente 8 personas")
        self.assertTrue(all(0 <= a.risk_aversion <= 100 for a in agents))
        self.assertTrue(all(0 <= a.price_sensitivity <= 100 for a in agents))

    def test_swarm_simulation_and_metrics(self):
        graph_rag = GraphRAG(llm=self.llm)
        graph_rag.build_from_text(self.scenario)
        gen = PersonaGenerator(llm=self.llm)
        agents = gen.generate_population(self.scenario, graph_rag, total_agents=6)

        engine = SwarmSimulationEngine(scenario=self.scenario, agents=agents, graph_rag=graph_rag, llm=self.llm)
        rounds = engine.run_simulation(total_rounds=2)

        self.assertEqual(len(rounds), 2, "Deveria executar 2 rodadas")
        self.assertGreater(len(engine.feed), 0, "Deveria ter gerado postagens no feed")
        self.assertTrue(-1.0 <= rounds[-1].average_sentiment <= 1.0)

    def test_prediction_report_and_interrogation(self):
        graph_rag = GraphRAG(llm=self.llm)
        graph_rag.build_from_text(self.scenario)
        gen = PersonaGenerator(llm=self.llm)
        agents = gen.generate_population(self.scenario, graph_rag, total_agents=6)

        engine = SwarmSimulationEngine(scenario=self.scenario, agents=agents, graph_rag=graph_rag, llm=self.llm)
        engine.run_simulation(total_rounds=2)

        reporter = ReportAgent(simulation=engine, llm=self.llm)
        report = reporter.generate_prediction_report()

        self.assertIsNotNone(report.raw_markdown)
        self.assertTrue(0 <= report.success_probability <= 100)

        # Teste de interrogação do oráculo
        answer = reporter.interrogate_world("Qual foi a principal crítica?")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 5)

        # Teste de entrevista com agente
        agent_reply = reporter.chat_with_agent(agents[0].id, "Você pretende cancelar o serviço?")
        self.assertIsInstance(agent_reply, str)
        self.assertGreater(len(agent_reply), 5)

    def test_document_store_lifecycle(self):
        store = DocumentStore()
        doc = store.add_document("teste_unit.txt", "Texto de teste para validação de repositório.", category="Teste")
        self.assertIn("id", doc)
        
        docs = store.list_documents()
        self.assertGreaterEqual(len(docs), 1)
        
        content = store.get_document_content(doc["id"])
        self.assertEqual(content, "Texto de teste para validação de repositório.")
        
        ctx = store.get_combined_context([doc["id"]])
        self.assertIn("teste_unit.txt", ctx)
        
        deleted = store.delete_document(doc["id"])
        self.assertTrue(deleted)


if __name__ == "__main__":
    unittest.main()

