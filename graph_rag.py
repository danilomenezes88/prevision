"""
Prevision AI Engine - GraphRAG Module
Extração estruturada de entidades, conexões e grafo de conhecimento relacional
a partir de documentos e cenários semente (Seed Materials).
"""

import json
import logging
from typing import Dict, List, Any, Optional
import networkx as nx
from pydantic import BaseModel, Field

from llm_client import LLMClient

logger = logging.getLogger("Prevision.GraphRAG")


class Entity(BaseModel):
    name: str
    type: str = "CONCEPT"  # STAKEHOLDER, COMPETITOR, REGULATOR, PRODUCT, DECISION_MAKER
    description: str = ""


class Relationship(BaseModel):
    source: str
    target: str
    relation: str
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.1, le=5.0)


class KnowledgeGraphResult(BaseModel):
    entities: List[Entity] = []
    relationships: List[Relationship] = []


class GraphRAG:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()
        self.graph = nx.DiGraph()

    def build_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extrai entidades e relacionamentos do texto semente utilizando LLM
        e popula o grafo NetworkX interno.
        """
        system_prompt = (
            "Você é um especialista em Inteligência de Grafos e GraphRAG. "
            "Sua tarefa é analisar o texto fornecido e extrair todas as entidades relevantes "
            "(empresas, pessoas, produtos, órgãos reguladores, concorrentes, grupos de consumidores) "
            "e os relacionamentos causais, comerciais ou de influência entre elas.\n"
            "Retorne OBRIGATORIAMENTE um JSON com as chaves 'entities' e 'relationships'.\n"
            "Formato:\n"
            "{\n"
            '  "entities": [{"name": "...", "type": "STAKEHOLDER|COMPETITOR|REGULATOR|PRODUCT|DECISION_MAKER", "description": "..."}],\n'
            '  "relationships": [{"source": "...", "target": "...", "relation": "...", "sentiment": -1.0 a 1.0, "weight": 1.0}]\n'
            "}"
        )

        user_prompt = f"Analise o seguinte cenário/documento semente e extraia o grafo:\n\n{text}"

        response_str = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            response_json=True
        )

        try:
            # Limpeza caso o LLM retorne blocos markdown ```json
            cleaned_str = response_str.strip()
            if cleaned_str.startswith("```json"):
                cleaned_str = cleaned_str[7:]
            if cleaned_str.startswith("```"):
                cleaned_str = cleaned_str[3:]
            if cleaned_str.endswith("```"):
                cleaned_str = cleaned_str[:-3]

            data = json.loads(cleaned_str.strip())
            kg_result = KnowledgeGraphResult(**data)
        except Exception as e:
            logger.warning(f"Erro ao decodificar JSON do grafo ({e}). Utilizando fallback seguro.")
            kg_result = KnowledgeGraphResult(
                entities=[
                    Entity(name="Consumidores", type="STAKEHOLDER", description="Base principal de usuários"),
                    Entity(name="Gestores da Decisão", type="DECISION_MAKER", description="Autores da proposta"),
                    Entity(name="Concorrentes do Setor", type="COMPETITOR", description="Alternativas de mercado")
                ],
                relationships=[
                    Relationship(source="Gestores da Decisão", target="Consumidores", relation="propõe_mudança", sentiment=-0.2),
                    Relationship(source="Consumidores", target="Concorrentes do Setor", relation="avalia_migração", sentiment=0.3)
                ]
            )

        # Atualiza a estrutura do NetworkX
        self.graph.clear()
        for ent in kg_result.entities:
            self.graph.add_node(ent.name, type=ent.type, description=ent.description)

        for rel in kg_result.relationships:
            self.graph.add_edge(
                rel.source,
                rel.target,
                relation=rel.relation,
                sentiment=rel.sentiment,
                weight=rel.weight
            )

        return self.to_dict()

    def get_context_for_persona(self, role: str) -> str:
        """Retorna os nós e relacionamentos do grafo mais relevantes para um determinado papel."""
        if self.graph.number_of_nodes() == 0:
            return "Sem dados prévios de grafo."

        context_lines = []
        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation", "relaciona-se com")
            sent = data.get("sentiment", 0.0)
            context_lines.append(f"- [{u}] --({rel}, sentimento={sent:+.1f})--> [{v}]")

        return "\n".join(context_lines[:15])

    def to_dict(self) -> Dict[str, Any]:
        """Exporta os dados do grafo em formato compatível com visualizadores (D3 / Streamlit / JSON)."""
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": node_id,
                "type": attrs.get("type", "CONCEPT"),
                "description": attrs.get("description", "")
            })

        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relation": attrs.get("relation", ""),
                "sentiment": attrs.get("sentiment", 0.0),
                "weight": attrs.get("weight", 1.0)
            })

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "nodes": nodes,
            "edges": edges
        }
