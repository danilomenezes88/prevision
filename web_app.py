"""
MiroFish BR - Web Application (Streamlit)
Painel Interativo de Simulação Preditiva e Inteligência de Enxame.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys

# Garante path local
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from llm_client import LLMClient
from graph_rag import GraphRAG
from persona_generator import PersonaGenerator
from simulation_engine import SwarmSimulationEngine
from report_agent import ReportAgent

# Configuração de Página
st.set_page_config(
    page_title="MiroFish BR - Simulação Preditiva",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização visual moderna
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
    }
    .post-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-pos {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .badge-neg {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Configurações
st.sidebar.title("⚙️ Configurações do Motor")
st.sidebar.markdown("Configure as credenciais e parâmetros da inteligência de enxame.")

provider = st.sidebar.selectbox(
    "Provedor de LLM:",
    ["Anthropic (Claude)", "OpenAI", "Gemini", "Groq", "Ollama"],
    index=0
)

# Mapeia nome do provedor
clean_provider = "anthropic" if "Anthropic" in provider else provider.lower()
config.llm_provider = clean_provider

# Chave pré-carregada padrão
default_key = config.anthropic_api_key if clean_provider == "anthropic" else ""
api_key = st.sidebar.text_input(
    f"Chave de API ({clean_provider.capitalize()}):",
    value=default_key,
    type="password",
    help="Chave carregada do ambiente. Altere se desejar usar outra."
)

if clean_provider == "anthropic" and (api_key or config.anthropic_api_key):
    config.anthropic_api_key = api_key or config.anthropic_api_key
    st.sidebar.success("⚡ Claude 3.5 / Haiku Conectado")
elif api_key:
    if clean_provider == "openai":
        config.openai_api_key = api_key
    elif clean_provider == "gemini":
        config.gemini_api_key = api_key
    elif clean_provider == "groq":
        config.groq_api_key = api_key

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros do Enxame")
num_agents = st.sidebar.slider("Número de Agentes (Personas):", min_value=6, max_value=40, value=15, step=1)
num_rounds = st.sidebar.slider("Número de Rodadas (Ticks):", min_value=1, max_value=5, value=3, step=1)

# Estado de Sessão
if "sim_data" not in st.session_state:
    st.session_state.sim_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cabeçalho Principal
st.markdown('<div class="main-header">🐟 MiroFish BR — Swarm Intelligence Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Motor de simulação preditiva multiagente e previsão social baseada em GraphRAG.</div>', unsafe_allow_html=True)

# Abas Principais
tab_input, tab_graph, tab_personas, tab_feed, tab_report, tab_qa = st.tabs([
    "🎯 1. Cenário Semente",
    "🕸️ 2. Grafo (GraphRAG)",
    "👥 3. Enxame de Agentes",
    "📱 4. Feed da Simulação",
    "📊 5. Relatório Preditivo",
    "💬 6. Interrogar o Mundo"
])

# ABA 1: CENÁRIO
with tab_input:
    st.subheader("Defina o Cenário a ser Previsto")
    
    presets = {
        "Customizado": "",
        "Reajuste de Preço de Streaming (+25%)": "A plataforma de streaming StreamFlix anunciou no Brasil um reajuste de 25% na mensalidade padrão e o início do bloqueio automático de compartilhamento de senhas fora da mesma residência, oferecendo apenas a opção de perfil extra pago por R$ 12,90/mês.",
        "Lançamento de Conta Digital com Taxa de Inatividade": "O banco digital NeoBank anunciou que contas com saldo menor que R$ 100 sem movimentação por mais de 90 dias passarão a pagar tarifa mensal de manutenção de R$ 9,90.",
        "Retorno 100% Presencial no Trabalho": "A empresa de tecnologia TechCorp anunciou o encerramento do modelo híbrido e a exigência de trabalho presencial compulsório de 5 dias por semana para todos os funcionários."
    }
    
    selected_preset = st.selectbox("Escolha um cenário pronto ou digite o seu:", list(presets.keys()))
    default_text = presets[selected_preset] if selected_preset != "Customizado" else "Digite o cenário de mercado, política ou produto aqui..."
    
    scenario_input = st.text_area("Descrição detalhada do cenário:", value=default_text, height=140)

    if st.button("🚀 Iniciar Simulação de Enxame", type="primary", use_container_width=True):
        if not scenario_input.strip() or scenario_input == "Digite o cenário de mercado, política ou produto aqui...":
            st.warning("Por favor, descreva um cenário válido.")
        else:
            with st.spinner("Processando GraphRAG e simulando enxame..."):
                llm = LLMClient()
                
                # 1. GraphRAG
                graph_rag = GraphRAG(llm=llm)
                graph_data = graph_rag.build_from_text(scenario_input)
                
                # 2. Personas
                gen = PersonaGenerator(llm=llm)
                agents = gen.generate_population(scenario_input, graph_rag, total_agents=num_agents)
                
                # 3. Engine
                engine = SwarmSimulationEngine(scenario=scenario_input, agents=agents, graph_rag=graph_rag, llm=llm)
                rounds_history = engine.run_simulation(total_rounds=num_rounds)
                
                # 4. Report
                reporter = ReportAgent(simulation=engine, llm=llm)
                report = reporter.generate_prediction_report()
                
                st.session_state.sim_data = {
                    "scenario": scenario_input,
                    "graph_data": graph_data,
                    "agents": agents,
                    "engine": engine,
                    "rounds_history": rounds_history,
                    "reporter": reporter,
                    "report": report
                }
                st.success("Simulação concluída com sucesso! Navegue pelas abas acima para explorar.")

# Verificação se simulação foi executada
data = st.session_state.sim_data

# ABA 2: GRAPHRAG
with tab_graph:
    if not data:
        st.info("Execute uma simulação na aba '1. Cenário Semente' para visualizar o grafo.")
    else:
        st.subheader("Entidades e Relacionamentos Extraídos (GraphRAG)")
        g = data["graph_data"]
        
        col1, col2 = st.columns(2)
        col1.metric("Nós (Entidades)", g["num_nodes"])
        col2.metric("Arestas (Relações)", g["num_edges"])
        
        col_nodes, col_edges = st.columns(2)
        with col_nodes:
            st.markdown("#### Entidades Identificadas")
            df_nodes = pd.DataFrame(g["nodes"])
            st.dataframe(df_nodes, use_container_width=True)
            
        with col_edges:
            st.markdown("#### Relações de Causa e Efeito")
            df_edges = pd.DataFrame(g["edges"])
            st.dataframe(df_edges, use_container_width=True)

# ABA 3: PERSONAS
with tab_personas:
    if not data:
        st.info("Execute uma simulação para visualizar a população de agentes.")
    else:
        st.subheader(f"População de Agentes ({len(data['agents'])} Personas)")
        agents = data["agents"]
        
        # Grid de cards de agentes
        cols = st.columns(3)
        for idx, agent in enumerate(agents):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 12px;">
                    <b>{agent.name}</b><br>
                    <small><b>Papel:</b> {agent.role}</small><br>
                    <small><b>Idade:</b> {agent.age} anos | <b>Renda:</b> {agent.income_level}</small><br>
                    <hr style="margin: 6px 0;">
                    <small>Preço: {agent.price_sensitivity}% | Risco: {agent.risk_aversion}% | Ceticismo: {agent.skepticism}%</small><br>
                    <small><b>Sentimento:</b> {agent.current_sentiment:+.2f} | <b>Postura:</b> {agent.stance}</small>
                </div>
                """, unsafe_allow_html=True)

# ABA 4: FEED
with tab_feed:
    if not data:
        st.info("Execute uma simulação para acompanhar as interações do enxame.")
    else:
        st.subheader("Feed Social Simulado")
        engine = data["engine"]
        
        round_filter = st.selectbox(
            "Filtrar por Rodada:",
            ["Todas"] + [f"Rodada {r}" for r in range(1, num_rounds + 1)]
        )
        
        posts = engine.feed
        if round_filter != "Todas":
            r_num = int(round_filter.split()[1])
            posts = [p for p in posts if p.round_num == r_num]
            
        for post in posts:
            badge_class = "badge-pos" if post.sentiment >= 0 else "badge-neg"
            badge_label = f"Sentimento: {post.sentiment:+.1f}"
            st.markdown(f"""
            <div class="post-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b>@{post.author_name}</b>
                    <span class="{badge_class}">{badge_label}</span>
                </div>
                <div style="color: #64748B; font-size: 0.85rem; margin-bottom: 6px;">
                    {post.author_role} • Rodada {post.round_num} • Ação: {post.action_type}
                </div>
                <div style="font-size: 0.95rem; margin-bottom: 8px;">
                    "{post.content}"
                </div>
                <div style="font-size: 0.85rem; color: #475569;">
                    ❤️ {post.likes} concordâncias
                </div>
            </div>
            """, unsafe_allow_html=True)

# ABA 5: RELATÓRIO PREDITIVO
with tab_report:
    if not data:
        st.info("Execute uma simulação para obter o relatório preditivo executivo.")
    else:
        rep = data["report"]
        rounds = data["rounds_history"]
        
        st.subheader("Relatório Preditivo de Tomada de Decisão")
        
        # Cartões de Métricas Principais
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probabilidade de Sucesso", f"{rep.success_probability:.1f}%")
        m2.metric("Nível de Risco", rep.risk_level)
        m3.metric("Polarização Social", rep.polarization_degree)
        final_sent = rounds[-1].average_sentiment if rounds else 0.0
        m4.metric("Sentimento Médio Final", f"{final_sent:+.2f}")
        
        st.markdown("---")
        
        # Gráficos de Evolução
        st.markdown("#### Evolução Temporal por Rodada")
        chart_data = pd.DataFrame({
            "Rodada": [r.round_num for r in rounds],
            "Sentimento Médio": [r.average_sentiment for r in rounds],
            "Índice de Polarização": [r.polarization_index for r in rounds],
            "Taxa de Boicote (%)": [r.boycott_rate for r in rounds]
        })
        st.line_chart(chart_data.set_index("Rodada"))
        
        st.markdown("---")
        st.markdown(rep.raw_markdown)
        
        st.download_button(
            label="📥 Baixar Relatório em Markdown",
            data=rep.raw_markdown,
            file_name="relatorio_preditivo_mirofish.md",
            mime="text/markdown"
        )

# ABA 6: INTERROGAR O MUNDO
with tab_qa:
    if not data:
        st.info("Execute uma simulação para habilitar a interrogação de agentes.")
    else:
        st.subheader("Oráculo & Entrevistas de Agentes")
        reporter = data["reporter"]
        
        mode = st.radio("Modo de Interação:", ["Oráculo Global (Sintetizador)", "Entrevistar Persona Específica"], horizontal=True)
        
        if mode == "Oráculo Global (Sintetizador)":
            user_question = st.text_input("Faça uma pergunta sobre os riscos, desdobramentos ou cenários alternativos:")
            if st.button("Consultar Oráculo"):
                if user_question:
                    with st.spinner("Analisando todas as interações do enxame..."):
                        ans = reporter.interrogate_world(user_question)
                        st.markdown(f"**Resposta da Simulação:**\n\n{ans}")
        else:
            agent_dict = {f"{a.name} ({a.role})": a.id for a in data["agents"]}
            chosen_agent_label = st.selectbox("Escolha o agente para entrevistar:", list(agent_dict.keys()))
            agent_id = agent_dict[chosen_agent_label]
            
            agent_question = st.text_input(f"Pergunte algo diretamente para {chosen_agent_label}:")
            if st.button("Enviar Pergunta ao Agente"):
                if agent_question:
                    with st.spinner("O agente está formulando sua resposta com base em suas convicções..."):
                        agent_reply = reporter.chat_with_agent(agent_id, agent_question)
                        st.markdown(f"**Resposta de {chosen_agent_label}:**\n\n{agent_reply}")
