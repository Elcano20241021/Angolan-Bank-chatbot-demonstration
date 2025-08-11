import os
from pathlib import Path
import re
import spacy
import unicodedata
from unidecode import unidecode
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from typing import List
import tiktoken # Importação para a contagem de tokens
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

# LangChain & Transformers
from openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#====================================================== Config & Env ====================================================================

load_dotenv()
os.environ["USER_AGENT"] = "chatbot-bancosol"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

if not OPENAI_API_KEY:
    st.error("Erro: A variável de ambiente OPENAI_API_KEY não está definida.")
    st.stop()

if not HUGGINGFACEHUB_API_TOKEN:
    st.error("Erro: A variável de ambiente HUGGINGFACEHUB_API_TOKEN não está definida.")
    st.stop()

# Parâmetros
PERSIST_DIRECTORY = "./chroma_db_bancosol"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHAT_HISTORY_LIMIT = 10 # Define o limite para o histórico de convers
TOKEN_LIMIT_CONTEXT = 2500
PDF_DATA_DIRECTORY = Path(
    os.getenv("PDF_DATA_DIRECTORY", r"C:\path\to\your\data_pdfs")
).resolve()
encoding = tiktoken.get_encoding("cl100k_base")

#====================================================== Sinónimos (normalização leve da query) ====================================================================
intent_synonyms = {
   "seguro de saude": "seguro vida",
    "app bancária": "solnet",
    "aplicativo do banco": "solnet",
    "banco no celular": "solnet",
    "abertura de conta": "abertura de conta_à_ordem_particular",
    "Tratar Cartão Multicaixa": "tratar do Cartão de Débito Multicaixa",
    "abrir conta para menor": "conta à ordem bankita",
    "conta para menor de idade": "conta à ordem bankita",
    "poupança com juro fixo": "depósito a prazo sol futuro",
    "crédito casa": "habitação",
    "empréstimo para casa": "habitação",
    "empréstimo para carro": "automóvel",
    "crédito automóvel": "automóvel",
    "enviar dinheiro para o exterior": "operações sobre o estrangeiro",
    "transferência internacional": "operações sobre o estrangeiro",
    "seguro de viagem": "seguro viagem",
    "seguro carro": "seguro automóvel",
    "conta com cartão": "conta à ordem particular",
    "conta normal": "conta à ordem particular",
    "salário": "salário sol",
    "dicas para poupar": "educacao financeira",
    "como posso poupar": "educacao financeira",
    "conselho financeiro": "educacao financeira"
}

STOPWORDS_PT = {"de", "da", "do", "das", "dos"}

def _norm(t) -> str:
    return re.sub(r"\s+", " ", unidecode("" if t is None else str(t)).lower().strip())

def _compile_synonym_patterns(synonyms: dict):
    compiled = []
    for raw_syn, raw_tgt in synonyms.items():
        syn = _norm(raw_syn)          # ex: "seguro de saude"
        tgt = _norm(raw_tgt)          # ex: "seguro vida"
        if not syn or not tgt:
            continue
        tokens = syn.split()
        parts = []
        for i, tok in enumerate(tokens):
            if tok in STOPWORDS_PT:
                parts.append(r"(?:\s+" + re.escape(tok) + r")?")   # torna "de/da/do..." opcional
            else:
                parts.append((r"\b" if i == 0 else r"\s+") + re.escape(tok))
        pattern = "".join(parts) + r"\b"
        rx = re.compile(pattern, re.IGNORECASE)
        compiled.append((rx, tgt, syn))
    # maiores primeiro para evitar matches parciais
    compiled.sort(key=lambda x: len(x[2]), reverse=True)
    return compiled

SYN_PATTERNS = _compile_synonym_patterns(intent_synonyms)

def translate_query(query: str) -> str:
    q = _norm(query)
    for rx, tgt, _ in SYN_PATTERNS:
        q = rx.sub(tgt, q)
    return q
#====================================================== RAG - Setup do Vector Store (PDFs locais) ====================================================================

# Setup do Vectorstore com Embeddings
@st.cache_resource(show_spinner="A preparar a base de conhecimento do Banco Sol...")
def setup_vector_store():
    all_processed_docs = []

    for root, dirs, files in os.walk(PDF_DATA_DIRECTORY):
        for file in files:
            if file.lower().endswith(".pdf"):
                filepath = os.path.join(root, file)
                loader = PyPDFLoader(filepath)
                docs = loader.load()

                # Metadados a partir do path
                relative_path = os.path.relpath(filepath, PDF_DATA_DIRECTORY)
                path_parts = relative_path.split(os.sep)

                metadata = {}
                if len(path_parts) >= 1:
                    metadata["categoria"] = path_parts[0]
                if len(path_parts) >= 2:
                    metadata["subcategoria"] = path_parts[1]
                if len(path_parts) >= 3:
                    metadata["produto"] = os.path.splitext(path_parts[2])[0].replace("_", " ").title()

                for doc in docs:
                    doc.metadata.update(metadata)
                    all_processed_docs.append(doc)

    if not all_processed_docs:
        st.error(f"Nenhum documento PDF encontrado no diretório: {PDF_DATA_DIRECTORY}")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", r"(?<=\.\s)(?=[A-ZÀ-Ú])", " ", ""],
        is_separator_regex=True,
        keep_separator=True,
        add_start_index=True
    )

    chunks = splitter.split_documents(all_processed_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
    )

    # Se já existir, reabre; caso contrário, cria e persiste
    if os.path.exists(PERSIST_DIRECTORY):
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name="bancosol_docs"
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name="bancosol_docs"
        )

    return vectorstore

#====================================================== Compactação de contexto (token-efficient) ====================================================================

def truncate_context_by_token_limit(documents, token_limit=TOKEN_LIMIT_CONTEXT, encoding=encoding):
    """
    Gera um contexto compacto a partir de uma lista de documentos:
    - Deduplica conteúdo (normaliza espaços e remove repetições).
    - Concatena até atingir o limite de tokens.
    - Aceita lista de Document (LangChain) ou strings.

    Parâmetros:
        documents (list): lista de Document ou strings.
        token_limit (int): limite máximo de tokens do contexto.
        encoding: codificador tiktoken já inicializado.

    Retorna:
        str: contexto concatenado e deduplicado, respeitando o limite de tokens.
    """
    if not documents:
        return ""

    seen = set()
    parts = []
    total_tokens = 0

    for d in documents:
        txt = d.page_content if hasattr(d, "page_content") else str(d)
        norm = " ".join(txt.split())
        if not norm or norm in seen:
            continue
        seen.add(norm)

        toks = encoding.encode(norm)
        n = len(toks)

        if total_tokens + n > token_limit:
            # adiciona parte que ainda cabe (opcional, para aproveitar o limite)
            remaining = token_limit - total_tokens
            if remaining > 0:
                partial = encoding.decode(toks[:remaining]).strip()
                if partial:
                    parts.append(partial)
            break

        parts.append(norm)
        total_tokens += n

    return "\n\n---\n\n".join(parts).strip()

#====================================================== System prompt ====================================================================

def generate_answer_openai(context: str, user_question: str) -> str:
    """
    Gera resposta usando apenas o contexto recuperado (RAG).
    - Responde somente sobre o Banco Sol.
    - Não inventa; se faltar contexto, orienta a contactar apoio.
    - Tom cordial e conciso; usa bullets quando útil.
    - Apenas UMA pergunta de continuidade quando fizer sentido.
    """
    try:
        # Prompt de sistema aprimorado para lidar com contexto e falta de informação
        system_prompt = (
           "Você é o 'Sol', assistente virtual oficial do Banco Sol. "
            "Responda SOMENTE sobre produtos e serviços do Banco Sol. "
            "Use EXCLUSIVAMENTE o contexto fornecido; não use conhecimento externo. "
            "Se a resposta não estiver no contexto, diga educadamente que não possui a informação "
            "e sugira contacto com o apoio ao cliente. "
            "Não invente detalhes. Seja claro, cordial e conciso. "
            "Use listas curtas quando ajudarem a leitura. "
            "Se fizer sentido, finalize com UMA pergunta natural para avançar a conversa; caso contrário, não pergunte. Após responder, inclua uma pergunta para aprofundar o interesse do cliente de forma natural, sem usar a mesma frase repetidamente. Adapte a pergunta ao contexto da interação e ao tópico. Se a pergunta for curta, como 'sim' ou 'não', use o histórico da conversa para inferir a intenção e dar uma resposta útil. Por exemplo:"
            " - Se a pergunta anterior foi 'Já tem um valor em mente para o seu empréstimo?' e a resposta for 'sim', pode perguntar: 'Qual o valor que tem em mente?'"
            " - Se o cliente perguntou sobre um depósito: 'Está a considerar poupar para algum objetivo específico?'"
            " - Se a pergunta for sobre um crédito: 'Já tem um valor em mente para o seu empréstimo?'"
            " - Se a pergunta for sobre um seguro: 'Para que finalidade seria o seguro? Por exemplo, proteção familiar, saúde ou automóvel?'"
            " - Se a pergunta for sobre uma conta: 'Qual a sua principal necessidade para uma conta? Talvez para gerir gastos diários, ou para poupança?'"
            "Finalize as suas respostas de forma natural e útil."
            
        )

        final_prompt = f"""
### Contexto:
{context}

### Pergunta:
{user_question}
""".strip()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.2,
            max_tokens=350
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[Erro com a API OpenAI: {e}]"

#====================================================== UI Streamlit ====================================================================

st.set_page_config(page_title="Assistente Sol", layout="wide")

mascote_path = "mascote_banco_sol.png"
if os.path.exists(mascote_path):
    mascote_img = Image.open(mascote_path)
    st.image(mascote_img, width=80)
st.title("Assistente Virtual Sol ☀️")

st.markdown("Bem-vindo! Eu sou o seu assistente virtual para ajudar com dúvidas sobre os produtos e serviços do Banco Sol.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Como posso ajudar você hoje?"}]

st.markdown("Para começar, escolha um tópico ou faça a sua pergunta diretamente:")
options = ["Abertura de Conta"]
initial_choice = st.selectbox("Tópicos frequentes:", options, index=None, placeholder="Escolha um tópico...")

vectorstore = setup_vector_store()
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input_from_chat = st.chat_input("Pergunte-me algo sobre o Banco Sol...")

if user_input_from_chat:
    user_input = user_input_from_chat
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response_text = ""
    
    with st.spinner("A pensar..."):
        # Lógica de escalonamento robusta
        escalation_keywords = ["agente", "humano", "ligar", "atendimento", "reclamação", "falar com"]
        if any(keyword in user_input.lower() for keyword in escalation_keywords):
            response_text = (
                "Entendido. Para questões que precisam de um contacto humano, pode ligar para a nossa linha de apoio ou visitar um dos nossos balcões."
                "\n\n**Linha de Apoio:** +244 923 120 210"
                "\n\n**Balcões:** Encontre o mais próximo aqui: [https://www.bancosol.ao/pt/balcoes](https://www.bancosol.ao/pt/balcoes)"
            )
        else:
            # Lógica para saudações simples
            greetings = ["olá", "oi", "bom dia", "boa tarde", "boa noite", "ola"]
            if any(g in user_input.lower().strip() for g in greetings) and len(user_input.split()) <= 3:
                response_text = "Olá! Como posso ajudar você hoje?"
            else:
                # Lógica principal: tradução, busca em RAG e geração de resposta
                translated_query = translate_query(user_input)
                context_docs = retriever.invoke(translated_query)
                context = truncate_context_by_token_limit(context_docs, token_limit=TOKEN_LIMIT_CONTEXT)
                response_text = generate_answer_openai(context, translated_query)
                
                # Se o contexto estiver vazio, o chatbot deve dar uma resposta padrão de falta de informação.
                if not context.strip():
                    response_text = (
                        "Desculpe, mas não encontrei informações sobre isso na minha base de dados."
                        "\n\nPara obter uma resposta detalhada, sugiro que entre em contacto com a nossa linha de apoio ou visite um dos nossos balcões."
                        "\n\n**Linha de Apoio:** +244 923 120 210"
                    )

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)
            
    # Limita o histórico de conversação
    if len(st.session_state.messages) > CHAT_HISTORY_LIMIT:
        st.session_state.messages = st.session_state.messages[-CHAT_HISTORY_LIMIT:]

elif initial_choice:
    user_input = initial_choice
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    response_text = ""

    with st.spinner("A pensar..."):
        translated_query = translate_query(user_input)
        context_docs = retriever.invoke(translated_query)
        context = truncate_context_by_token_limit(context_docs, token_limit=TOKEN_LIMIT_CONTEXT)
        response_text = generate_answer_openai(context, translated_query)
        
        if not context.strip():
            response_text = (
                "Desculpe, mas não encontrei informações sobre isso na minha base de dados."
                "\n\nPara obter uma resposta detalhada, sugiro que entre em contacto com a nossa linha de apoio ou visite um dos nossos balcões."
                "\n\n**Linha de Apoio:** +244 923 120 210"
            )
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)
        
    if len(st.session_state.messages) > CHAT_HISTORY_LIMIT:
        st.session_state.messages = st.session_state.messages[-CHAT_HISTORY_LIMIT:]
