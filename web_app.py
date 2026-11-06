"""
Prevision AI - Web Application (Streamlit)
Plataforma de Inteligência Preditiva, Biblioteca de Dados e Simulação de Enxame Multiagente.
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
from document_store import document_store

# Configuração de Página
st.set_page_config(
    page_title="Prevision - Inteligência Preditiva",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização visual moderna
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
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
    .repo-card {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Configurações
st.sidebar.title("⚙️ Configurações do Motor")
st.sidebar.markdown("Gerencie o provedor de IA e os parâmetros da inteligência de enxame.")

provider = st.sidebar.selectbox(
    "Provedor de LLM:",
    ["Anthropic (Claude)", "OpenAI", "Gemini", "Groq", "Ollama"],
    index=0
)

clean_provider = "anthropic" if "Anthropic" in provider else provider.lower()
config.llm_provider = clean_provider

default_key = config.anthropic_api_key if clean_provider == "anthropic" else ""
api_key = st.sidebar.text_input(
    f"Chave de API ({clean_provider.capitalize()}):",
    value=default_key,
    type="password",
    help="Chave carregada do ambiente seguro."
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
if "scenario_text" not in st.session_state:
    st.session_state.scenario_text = ""

# Cabeçalho Principal
st.markdown('<div class="main-header">🔮 Prevision — Swarm Intelligence Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Plataforma preditiva multiagente, GraphRAG relacional e Biblioteca de Conhecimento.</div>', unsafe_allow_html=True)

# Abas Principais
tab_input, tab_repo, tab_graph, tab_personas, tab_feed, tab_report, tab_qa = st.tabs([
    "🎯 1. Cenário & Simulação",
    "📚 2. Biblioteca de Dados",
    "🕸️ 3. Grafo (GraphRAG)",
    "👥 4. Enxame de Agentes",
    "📱 5. Feed da Simulação",
    "📊 6. Relatório Preditivo",
    "💬 7. Interrogar o Mundo"
])

# ==============================================================================
# ABA 1: CENÁRIO & SIMULAÇÃO
# ==============================================================================
with tab_input:
    st.subheader("1. Defina a Decisão ou Cenário a ser Previsto")

    col_mode, col_preset = st.columns([1, 2])
    with col_mode:
        input_mode = st.radio(
            "Origem do Cenário:",
            ["✍️ Digitar Cenário Próprio", "📋 Usar Exemplo Pronto"],
            horizontal=False
        )

    presets = {
        "Reajuste de Preço de Streaming (+25%)": "A plataforma de streaming StreamFlix anunciou no Brasil um reajuste de 25% na mensalidade padrão e o início do bloqueio automático de compartilhamento de senhas fora da mesma residência, oferecendo apenas a opção de perfil extra pago por R$ 12,90/mês.",
        "Lançamento de Conta Digital com Taxa de Inatividade": "O banco digital NeoBank anunciou que contas com saldo menor que R$ 100 sem movimentação por mais de 90 dias passarão a pagar tarifa mensal de manutenção de R$ 9,90.",
        "Retorno 100% Presencial no Trabalho": "A empresa de tecnologia TechCorp anunciou o encerramento do modelo híbrido e a exigência de trabalho presencial compulsório de 5 dias por semana para todos os funcionários."
    }

    with col_preset:
        if input_mode == "📋 Usar Exemplo Pronto":
            selected_preset = st.selectbox("Escolha um cenário de exemplo:", list(presets.keys()))
            if st.button("📥 Inserir este exemplo no editor"):
                st.session_state.scenario_text = presets[selected_preset]
                st.rerun()

    scenario_input = st.text_area(
        "Descrição da decisão, política, lançamento de produto ou evento:",
        value=st.session_state.scenario_text,
        placeholder="Digite livremente o cenário que você deseja prever... Exemplo: 'Nossa empresa planeja lançar uma taxa de entrega de R$ 4,99 para pedidos abaixo de R$ 50...'",
        height=160,
        help="Escreva o contexto da decisão para os agentes debaterem e avaliarem."
    )
    st.session_state.scenario_text = scenario_input

    # SEÇÃO DA BIBLIOTECA DE DADOS NA SIMULAÇÃO
    st.markdown("---")
    st.subheader("2. Biblioteca de Dados (Repositório de Conhecimento)")

    all_docs = document_store.list_documents()

    if not all_docs:
        st.info("ℹ️ Nenhum documento cadastrado na biblioteca ainda. Você pode adicionar documentos (PDF, TXT, CSV) na aba **'📚 2. Biblioteca de Dados'** ou continuar apenas com o texto digitado.")
        selected_doc_ids = []
    else:
        st.markdown("Selecione quais documentos do seu repositório devem ser injetados para enriquecer o contexto dos agentes e do Grafo:")
        
        col_repo_mode, col_repo_select = st.columns([1, 2])
        with col_repo_mode:
            repo_selection_mode = st.radio(
                "Modo de uso dos dados:",
                ["🔘 Selecionar Manualmente", "⚡ Seleção Automática Relevante", "❌ Não usar dados do repositório"],
                index=0
            )

        with col_repo_select:
            if repo_selection_mode == "🔘 Selecionar Manualmente":
                doc_options = {d["id"]: f"📄 {d['name']} ({d['category']} • {d['word_count']} palavras)" for d in all_docs}
                selected_doc_ids = st.multiselect(
                    "Escolha os documentos para esta simulação:",
                    options=list(doc_options.keys()),
                    format_func=lambda x: doc_options[x],
                    default=list(doc_options.keys())[:3]
                )
            elif repo_selection_mode == "⚡ Seleção Automática Relevante":
                if scenario_input.strip():
                    auto_ids = document_store.auto_select_relevant_docs(scenario_input)
                    selected_doc_ids = auto_ids if auto_ids else [d["id"] for d in all_docs[:2]]
                    st.success(f"🤖 O motor identificou {len(selected_doc_ids)} documento(s) com palavras-chave relevantes ao seu cenário.")
                else:
                    selected_doc_ids = [d["id"] for d in all_docs[:2]]
                    st.caption("Digite o cenário para o motor buscar documentos por afinidade temática.")
            else:
                selected_doc_ids = []

        if selected_doc_ids:
            total_chars = sum([d["char_count"] for d in all_docs if d["id"] in selected_doc_ids])
            st.success(f"✓ **{len(selected_doc_ids)} documento(s) do repositório serão incorporados** ao GraphRAG e à memória dos agentes (~{total_chars:,} caracteres).")

    st.markdown("---")

    # BOTÃO PRINCIPAL DE EXECUÇÃO
    if st.button("🚀 Iniciar Simulação Preditiva", type="primary", use_container_width=True):
        if not scenario_input or not scenario_input.strip():
            st.warning("⚠️ Por favor, preencha o cenário no campo de texto acima.")
        else:
            with st.spinner("Construindo GraphRAG, instanciando enxame e simulando rodadas com Claude..."):
                llm = LLMClient()
                
                # Monta contexto combinado (Cenário + Documentos da Biblioteca)
                combined_context = scenario_input
                if selected_doc_ids:
                    repo_context = document_store.get_combined_context(selected_doc_ids)
                    combined_context = f"{scenario_input}\n\n[DADOS E DOCUMENTOS DA BIBLIOTECA DE CONHECIMENTO]:\n{repo_context}"

                # 1. GraphRAG
                graph_rag = GraphRAG(llm=llm)
                graph_data = graph_rag.build_from_text(combined_context)
                
                # 2. Personas
                gen = PersonaGenerator(llm=llm)
                agents = gen.generate_population(combined_context, graph_rag, total_agents=num_agents)
                
                # 3. Engine de Enxame
                engine = SwarmSimulationEngine(scenario=combined_context, agents=agents, graph_rag=graph_rag, llm=llm)
                rounds_history = engine.run_simulation(total_rounds=num_rounds)
                
                # 4. Relatório Preditivo
                reporter = ReportAgent(simulation=engine, llm=llm)
                report = reporter.generate_prediction_report()
                
                st.session_state.sim_data = {
                    "scenario": scenario_input,
                    "selected_docs": selected_doc_ids,
                    "graph_data": graph_data,
                    "agents": agents,
                    "engine": engine,
                    "rounds_history": rounds_history,
                    "reporter": reporter,
                    "report": report
                }
                st.success("✅ Simulação concluída com sucesso! Navegue pelas abas acima para conferir os resultados.")

# ==============================================================================
# ABA 2: BIBLIOTECA DE DADOS (REPOSITÓRIO)
# ==============================================================================
with tab_repo:
    st.subheader("📚 Repositório de Documentos e Biblioteca de Conhecimento")
    st.markdown(
        "Carregue relatórios, pesquisas de mercado, políticas de preços, históricos de clientes e documentos regulatórios. "
        "Eles ficam salvos no seu repositório permanente para você reutilizar em qualquer previsão sem precisar reenviar."
    )

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_files = st.file_uploader(
            "Enviar novos arquivos para o repositório (PDF, TXT, MD, CSV, JSON):",
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True
        )
    with col_up2:
        category_choice = st.selectbox(
            "Categoria dos documentos:",
            ["Pesquisa de Mercado", "Financeiro / Preços", "Políticas & Regras", "Feedback de Clientes", "Concorrência", "Geral"]
        )
        if uploaded_files:
            if st.button("💾 Salvar no Repositório", type="primary"):
                saved_count = 0
                for uf in uploaded_files:
                    file_bytes = uf.read()
                    extracted_text = document_store.extract_text_from_upload(uf.name, file_bytes)
                    if extracted_text and not extracted_text.startswith("[Erro"):
                        document_store.add_document(uf.name, extracted_text, category=category_choice)
                        saved_count += 1
                st.success(f"✓ {saved_count} documento(s) salvos com sucesso na biblioteca!")
                st.rerun()

    st.markdown("---")
    st.subheader("Documentos Salvos na Biblioteca")

    current_docs = document_store.list_documents()

    if not current_docs:
        st.info("Sua biblioteca de dados está vazia no momento.")
        if st.button("📥 Inserir Documentos de Exemplo (Demonstração)"):
            document_store.add_document(
                "Pesquisa_Mercado_Consumidor_2026.txt",
                "PESQUISA DE MERCADO NACIONAL 2026:\n- 72% dos consumidores brasileiros declaram que cancelariam um serviço caso haja aumento acima de 15% sem novas funcionalidades claras.\n- 58% dos clientes valorizam suporte ágil e estabilidade acima de inovações complexas.\n- O segmento jovem (18-34) apresenta a maior taxa de volatilidade e migração para novos concorrentes.",
                category="Pesquisa de Mercado"
            )
            document_store.add_document(
                "Analise_Concorrencia_e_Precos.txt",
                "MAPA DE CONCORRÊNCIA E PRECIFICAÇÃO:\n- Concorrente Alpha: Oferece planos a partir de R$ 29,90 com política agressiva de atração de clientes insatisfeitos.\n- Concorrente Beta: Foco em clientes premium com taxa de retenção de 85%.\n- Práticas recomendadas: Oferecer descontos progressivos para retenção de clientes antigos (grandfathering).",
                category="Concorrência"
            )
            document_store.add_document(
                "Diretrizes_Regulatorias_e_Procon.txt",
                "DIRETRIZES DE DEFESA DO CONSUMIDOR (PROCON):\n- Reajustes unilaterais exigem notificação prévia expressa de no mínimo 30 dias com transparência nos canais de atendimento.\n- Cláusulas de fidelidade automática pós-reajuste são consideradas abusivas por tribunais de pequenas causas.",
                category="Políticas & Regras"
            )
            st.success("Documentos de exemplo carregados!")
            st.rerun()
    else:
        for doc in current_docs:
            with st.container():
                st.markdown(f"""
                <div class="repo-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b>📄 {doc['name']}</b>
                        <span class="badge-pos">{doc['category']}</span>
                    </div>
                    <small style="color: #64748B;">
                        Criado em: {doc['created_at']} • {doc['char_count']:,} caracteres • ~{doc['word_count']:,} palavras
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                col_view, col_del = st.columns([5, 1])
                with col_view:
                    with st.expander(f"Ver prévia do conteúdo ({doc['name']})"):
                        content = document_store.get_document_content(doc["id"]) or ""
                        st.text_area("Conteúdo extraído:", value=content, height=180, key=f"preview_{doc['id']}", disabled=True)
                with col_del:
                    if st.button("🗑️ Excluir", key=f"del_{doc['id']}"):
                        document_store.delete_document(doc["id"])
                        st.rerun()

# ==============================================================================
# DADOS DA SIMULAÇÃO
# ==============================================================================
data = st.session_state.sim_data

# ==============================================================================
# ABA 3: GRAPHRAG
# ==============================================================================
with tab_graph:
    if not data:
        st.info("Execute uma simulação na aba '1. Cenário & Simulação' para visualizar o Grafo de Conhecimento.")
    else:
        st.subheader("Entidades e Relacionamentos Extraídos (GraphRAG)")
        g = data["graph_data"]
        
        col1, col2 = st.columns(2)
        col1.metric("Nós (Entidades Mapeadas)", g["num_nodes"])
        col2.metric("Arestas (Relações Causais)", g["num_edges"])
        
        col_nodes, col_edges = st.columns(2)
        with col_nodes:
            st.markdown("#### Entidades Identificadas no Cenário e Documentos")
            df_nodes = pd.DataFrame(g["nodes"])
            st.dataframe(df_nodes, use_container_width=True)
            
        with col_edges:
            st.markdown("#### Relações de Causa, Efeito e Influência")
            df_edges = pd.DataFrame(g["edges"])
            st.dataframe(df_edges, use_container_width=True)

# ==============================================================================
# ABA 4: PERSONAS
# ==============================================================================
with tab_personas:
    if not data:
        st.info("Execute uma simulação para visualizar a população de agentes.")
    else:
        st.subheader(f"População Heterogênea de Agentes ({len(data['agents'])} Personas)")
        agents = data["agents"]
        
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

# ==============================================================================
# ABA 5: FEED SOCIAL SIMULADO
# ==============================================================================
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

# ==============================================================================
# ABA 6: RELATÓRIO PREDITIVO
# ==============================================================================
with tab_report:
    if not data:
        st.info("Execute uma simulação para obter o relatório preditivo executivo.")
    else:
        rep = data["report"]
        rounds = data["rounds_history"]
        
        st.subheader("Relatório Preditivo de Tomada de Decisão")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probabilidade de Sucesso", f"{rep.success_probability:.1f}%")
        m2.metric("Nível de Risco", rep.risk_level)
        m3.metric("Polarização Social", rep.polarization_degree)
        final_sent = rounds[-1].average_sentiment if rounds else 0.0
        m4.metric("Sentimento Médio Final", f"{final_sent:+.2f}")
        
        st.markdown("---")
        
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
            file_name="relatorio_preditivo_prevision.md",
            mime="text/markdown"
        )

# ==============================================================================
# ABA 7: INTERROGAR O MUNDO
# ==============================================================================
with tab_qa:
    if not data:
        st.info("Execute uma simulação para habilitar a interrogação de agentes.")
    else:
        st.subheader("Oráculo Analítico & Entrevistas com Agentes")
        reporter = data["reporter"]
        
        mode = st.radio("Modo de Interação:", ["Oráculo Global (Sintetizador)", "Entrevistar Persona Específica"], horizontal=True)
        
        if mode == "Oráculo Global (Sintetizador)":
            user_question = st.text_input("Faça uma pergunta sobre os riscos, desdobramentos ou cenários alternativos:")
            if st.button("Consultar Oráculo"):
                if user_question:
                    with st.spinner("Analisando evidências do enxame com Claude..."):
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
