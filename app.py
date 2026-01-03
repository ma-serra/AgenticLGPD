import os
import json
import re
import faiss
import numpy as np
from typing_extensions import TypedDict
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END
import gradio as gr

# ---------------------------
# Variáveis de ambiente
# ---------------------------
api_key = os.environ.get("OPENAI_API_KEY")

# ---------------------------
# Carregar modelo de embeddings
# ---------------------------
model_name = "sentence-transformers/all-mpnet-base-v2"
try:
    embed_model = SentenceTransformer(model_name)
    embed_model_error = None
except Exception as exc:
    embed_model = None
    embed_model_error = str(exc)

# ---------------------------
# Funções de recuperação de documentos
# ---------------------------
def create_bm25_index(documentos):
    tokenized_chunks = [re.findall(r'\w+', doc['content'].lower()) for doc in documentos]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, tokenized_chunks

def hybrid_search(query, documentos, bm25, tokenized_chunks, faiss_index, embed_model, top_k_sem=5, top_k_bm25=5):
    query_emb = embed_model.encode([query]).astype("float32")
    D, I = faiss_index.search(query_emb, top_k_sem)
    semantic_docs = [documentos[i] for i in I[0]]
    query_tokens = re.findall(r'\w+', query.lower())
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]
    exact_docs = [documentos[i] for i in bm25_indices]
    seen_ids = set()
    resultados = []
    for doc in semantic_docs + exact_docs:
        doc_id = doc["metadata"].get("id", hash(doc["content"]))
        if doc_id not in seen_ids:
            resultados.append({"texto": doc["content"], "metadados": doc["metadata"]})
            seen_ids.add(doc_id)
    return resultados

# ---------------------------
# Tipagem do estado do agente
# ---------------------------
class AgentState(TypedDict):
    question: str
    answer: str
    topic: str
    documents_lei: list
    documents_jurisprudencia: list
    memory: list
    
# ---------------------------------------------
# PRÉ-CARREGAMENTO DE DADOS E ÍNDICES (OTIMIZAÇÃO)
# ---------------------------------------------
chunks_lei = []
chunks_jurisprudencia = []
bm25_lei = bm25_jurisprudencia = None
tokenized_chunks_lei = tokenized_chunks_jurisprudencia = []
index_lei = index_jurisprudencia = None
resources_ready = False
resources_error = None

try:
    with open("lei_chunks_com_metadados_lei.json", "r", encoding="utf-8") as f:
        chunks_lei = json.load(f)
    index_lei = faiss.read_index("lei_faiss_lei.index")
    bm25_lei, tokenized_chunks_lei = create_bm25_index(chunks_lei)

    with open("lei_chunks_com_metadados_jurisprudencia.json", "r", encoding="utf-8") as f:
        chunks_jurisprudencia = json.load(f)
    index_jurisprudencia = faiss.read_index("lei_faiss_jurisprudencia.index")
    bm25_jurisprudencia, tokenized_chunks_jurisprudencia = create_bm25_index(chunks_jurisprudencia)
    resources_ready = True
except FileNotFoundError as exc:
    resources_error = f"Arquivos de dados ausentes: {exc}"
except Exception as exc:
    resources_error = str(exc)

# ---------------------------
# Funções do agente
# ---------------------------
def check_question(state):
    system_prompt = """Você é um avaliador especializado em proteção de dados pessoais. Sua tarefa é verificar se a pergunta feita pelo usuário está relacionada à LGPD ou jurisprudência.
    Responda com: "Lei", "Jurisprudencia", "Lei,Jurisprudencia" ou "False"."""
    TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User question: {question}"),
    ])
    prompt = TEMPLATE.format(question=state['question'])
    model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    response_text = model.invoke(prompt)
    state['topic'] = response_text.content.strip()
    return state

def topic_router(state):
    topic = state['topic']
    if topic == "Lei":
        return "retrieve_docs_lei"
    elif topic == "Jurisprudencia":
        return "retrieve_docs_jurisprudencia"
    elif topic == "Lei,Jurisprudencia":
        return "retrieve_docs_lei_jurisprudencia"
    else:
        return "off_topic_response"

def off_topic_response(state):
    state['answer'] = "Desculpe, só posso esclarecer dúvidas relacionadas à LGPD."
    return state

def retrieve_docs_lei(state):
    # Usa os dados carregados globalmente
    docs_faiss = hybrid_search(state['question'], chunks_lei, bm25_lei, tokenized_chunks_lei, index_lei, embed_model)
    state['documents_lei'] = [doc["texto"] for doc in docs_faiss]
    return state

def retrieve_docs_jurisprudencia(state):
    # Usa os dados carregados globalmente
    docs_faiss = hybrid_search(state['question'], chunks_jurisprudencia, bm25_jurisprudencia, tokenized_chunks_jurisprudencia, index_jurisprudencia, embed_model)
    state['documents_jurisprudencia'] = [doc["texto"] for doc in docs_faiss]
    return state

def retrieve_docs_lei_jurisprudencia(state):
    state = retrieve_docs_lei(state)
    state = retrieve_docs_jurisprudencia(state)
    return state

def generate(state):
    all_docs = state.get('documents_lei', []) + state.get('documents_jurisprudencia', [])
    system_prompt = """Você é um assistente especializado em LGPD, responsável por responder às perguntas dos usuários sobre proteção de dados pessoais. Responda de forma clara e objetiva, evitando ser excessivamente detalhado ou muito breve. Não se refira a si mesmo como "assistente" ou mencione seu papel na resposta."""
    TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Contexto: {documents}\nHistórico da conversa: {memory}\nPergunta: {question}")
    ])
    prompt = TEMPLATE.format(documents=all_docs, memory=state['memory'], question=state['question'])
    model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    response_text = model.invoke(prompt)
    state['answer'] = response_text.content.strip()
    return state

def improve_answer(state):
    system = """Como assistente de LGPD, revise e aprimore a resposta a uma pergunta do usuário. Sua tarefa é:
    - Garantir que a resposta seja adequada, clara e informativa.
    - Editar ou remover partes da resposta conforme necessário.
    - Manter um tom educado, profissional e atencioso.
    - Fornecer apenas a resposta aprimorada, sem frases introdutórias ou comentários.
    - Concluir a resposta com uma pergunta aberta para incentivar novas dúvidas ou esclarecimentos adicionais.
    - Considerar o histórico da conversa para tornar a resposta mais relevante e útil.
    - Incluir quebras de linha (\n) ao final de cada frase ou ponto lógico."""
    TEMPLATE = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Pergunta: {question}\nHistórico: {memory}\nResposta inicial: {answer}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    prompt = TEMPLATE.format(question=state['question'], memory=state['memory'], answer=state['answer'])
    response_text = model.invoke(prompt)
    state['answer'] = response_text.content.strip()
    return state

# ---------------------------
# Construção do workflow
# ---------------------------
workflow = StateGraph(AgentState)
workflow.add_node("check_question", check_question)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve_docs_lei", retrieve_docs_lei)
workflow.add_node("retrieve_docs_jurisprudencia", retrieve_docs_jurisprudencia)
workflow.add_node("retrieve_docs_lei_jurisprudencia", retrieve_docs_lei_jurisprudencia)
workflow.add_node("generate", generate)
workflow.add_node("improve_answer", improve_answer)

workflow.set_entry_point("check_question")
workflow.add_conditional_edges("check_question", topic_router, {
    "retrieve_docs_lei": "retrieve_docs_lei",
    "retrieve_docs_jurisprudencia": "retrieve_docs_jurisprudencia",
    "retrieve_docs_lei_jurisprudencia": "retrieve_docs_lei_jurisprudencia",
    "off_topic_response": "off_topic_response"
})
workflow.add_edge("retrieve_docs_lei", "generate")
workflow.add_edge("retrieve_docs_jurisprudencia", "generate")
workflow.add_edge("retrieve_docs_lei_jurisprudencia", "generate")
workflow.add_edge("generate", "improve_answer")
workflow.add_edge("improve_answer", END)
workflow.add_edge("off_topic_response", END)

app = workflow.compile()

# ---------------------------
# Função Gradio
# ---------------------------
def hf_chat(user_input, history):
    # Limites para controle de tokens
    MAX_INPUT_LENGTH = 1000
    MAX_MEMORY_TURNS = 5

    if not api_key:
        return "Defina a variável de ambiente OPENAI_API_KEY para usar o agente."

    if embed_model is None:
        return f"Não foi possível carregar o modelo de embeddings ({embed_model_error}). Verifique os requisitos antes de implantar."

    if not resources_ready:
        return "Os índices e arquivos de dados não estão disponíveis no servidor. Inclua os arquivos *.json e *.index antes de iniciar."

    # 1. Checagem do tamanho da entrada do usuário
    if len(user_input) > MAX_INPUT_LENGTH:
        return "Desculpe, sua pergunta é muito longa. Por favor, seja mais conciso para que eu possa ajudá-lo."
    
    # 2. Construção da memória com um limite de turnos
    memory = []
    # O Gradio já cuida da passagem e atualização do 'history'
    # Você só precisa usar o histórico para construir o 'memory'
    for question, answer_mem in history[-MAX_MEMORY_TURNS:]:
        if question:
            memory.append(f"Pergunta: {question}")
        if answer_mem:
            memory.append(f"Resposta: {answer_mem}")

    # Inicializa o estado com a pergunta e a memória para o workflow
    state = {
        "question": user_input,
        "memory": memory,
    }

    # Invoca o workflow com o estado atual
    final_state = app.invoke(state, {"recursion_limit": 50})
    answer = final_state.get("answer", "Desculpe, não consegui gerar uma resposta.")
    
    # Apenas retorna a resposta. O Gradio se encarrega do resto.
    return answer

# ---------------------------
# Interface Gradio
# ---------------------------
iface = gr.ChatInterface(
    fn=hf_chat,
    title="Agentic LGPD",
    description="Agente especializado em LGPD e jurisprudência. Digite sua pergunta abaixo."
)

iface.launch()
