from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import json
import docx
import PyPDF2
import os
import logging
import tiktoken
import hashlib
import re
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import time
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carregar credenciais do Streamlit Secrets
try:
    service_account_info = json.loads(st.secrets["google_service_account"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project_id = service_account_info["project_id"]
    logger.info("Credenciais carregadas do Streamlit Secrets")
except Exception as e:
    logger.error(f"Erro ao carregar credenciais do Streamlit Secrets: {e}")
    st.error("Erro ao carregar credenciais. Por favor, verifique a configuração dos secrets.")
    st.stop()

# Se as credenciais expiraram, atualize-as
if credentials.expired and credentials.refresh_token:
    try:
        credentials.refresh(Request())
        logger.info("Credenciais atualizadas com sucesso")
    except Exception as e:
        logger.error(f"Erro ao atualizar credenciais: {e}")
        st.error("Erro ao atualizar credenciais. Por favor, tente novamente mais tarde.")
        st.stop()

# Inicializar o modelo Gemini com as credenciais carregadas
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, credentials=credentials)
    logger.info("Modelo Gemini inicializado com sucesso")
except Exception as e:
    logger.error(f"Erro ao inicializar o modelo Gemini: {e}")
    st.error("Erro ao inicializar o modelo de IA. Por favor, tente novamente mais tarde.")
    st.stop()

# Função para contar tokens
def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Função para contar caracteres
def count_characters(text):
    return len(text)

# Funções para carregar e processar arquivos (JSON, DOCX, PDF)
# ... (manter as funções load_json, load_docx, load_pdf como estavam)

# Função para carregar todos os arquivos na pasta materiais
def load_materials(directory='materiais'):
    materials = []
    total_tokens = 0
    total_chars = 0
    if not os.path.exists(directory):
        logger.warning(f"Pasta de materiais não encontrada: {directory}")
        return "", 0, 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if filename.endswith('.json'):
                content = load_json(filepath)
            elif filename.endswith('.docx'):
                content = load_docx(filepath)
            elif filename.endswith('.pdf'):
                content = load_pdf(filepath)
            else:
                continue

            materials.append(str(content))
            content_str = str(content)
            total_tokens += num_tokens_from_string(content_str)
            total_chars += count_characters(content_str)
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo {filename}: {e}")

    materials_text = "\n\n".join(materials)
    logger.info(f"Total de tokens nos materiais: {total_tokens}")
    logger.info(f"Total de caracteres nos materiais: {total_chars}")
    return materials_text, total_tokens, total_chars

# Contexto fixo do agente
agent_context = (
    "Você é um agente inteligente e consultor comercial da empresa Sucesso em Vendas. "
    "Gostaria que me respondesse de forma objetiva e concisa, com uma explicação sobre e em seguida uma abordagem pratica de como fazer para resolver. "
    "Seu papel é fornecer assistência especializada utilizando o método de vendas da Sucesso em Vendas e ajudar com conselhos comerciais para gerentes, coordenadores e vendedores."
)

# Função para gerar a resposta
def generate_response(user_input, context):
    cache_key = hashlib.md5((user_input + context[:100]).encode()).hexdigest()
    
    if cache_key in st.session_state.response_cache:
        logger.info("Resposta encontrada no cache")
        return st.session_state.response_cache[cache_key]

    prompt = f"{context}\n\nUsuário: {user_input}\nChatbot:"
    input_tokens = num_tokens_from_string(prompt)
    input_chars = count_characters(prompt)
    logger.info(f"Tokens na entrada: {input_tokens}")
    logger.info(f"Caracteres na entrada: {input_chars}")
    
    model = ChatPromptTemplate.from_template(prompt) | llm
    try:
        response = model.invoke({'input': prompt})
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        response_tokens = num_tokens_from_string(response_content)
        response_chars = count_characters(response_content)
        logger.info(f"Tokens na resposta: {response_tokens}")
        logger.info(f"Caracteres na resposta: {response_chars}")
        
        total_tokens = input_tokens + response_tokens
        total_chars = input_chars + response_chars
        logger.info(f"Total de tokens nesta interação: {total_tokens}")
        logger.info(f"Total de caracteres nesta interação: {total_chars}")
        
        st.session_state.response_cache[cache_key] = response_content
        
        return response_content
    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {str(e)}")
        return "Ocorreu um erro ao gerar a resposta. Por favor, tente novamente."

# Função para exibir a resposta gradualmente
def display_typing_response(response_text, container):
    typing_speed = 0.01
    typed_text = ""
    for char in response_text:
        typed_text += char
        container.markdown(f"<div class='agent-message'>{typed_text}</div>", unsafe_allow_html=True)
        time.sleep(typing_speed)

# Função para extrair título do chat
def extract_title(message):
    words = re.findall(r'\b\w+\b', message)
    if len(words) >= 2:
        return f"{words[0]} {words[1]}..."
    return "Novo Chat"

# Configurar Streamlit
st.set_page_config(page_title='Consultor da Sucesso em Vendas', layout="wide")

# Adicionar estilos CSS personalizados
# ... (manter os estilos CSS como estavam)

# Carregar materiais e definir contexto
with st.spinner("Carregando materiais..."):
    try:
        materials_context, materials_tokens, materials_chars = load_materials()
        logger.info(f"Materiais carregados com sucesso. Total de tokens: {materials_tokens}, Total de caracteres: {materials_chars}")
    except Exception as e:
        materials_context = f"Erro ao carregar materiais: {e}"
        materials_tokens = 0
        materials_chars = 0
        logger.error(f"Erro ao carregar materiais: {e}")

context = f"{agent_context}\n\n{materials_context}"
context_tokens = num_tokens_from_string(context)
context_chars = count_characters(context)
logger.info(f"Total de tokens no contexto completo: {context_tokens}")
logger.info(f"Total de caracteres no contexto completo: {context_chars}")

# Inicializar o estado da sessão
if 'chats' not in st.session_state:
    current_date = datetime.now().strftime("%d/%m/%Y")
    st.session_state.chats = {'chat_1': {'date': current_date, 'messages': [], 'title': "Novo Chat"}}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = 'chat_1'
if 'user_interactions' not in st.session_state:
    st.session_state.user_interactions = 0
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'total_characters' not in st.session_state:
    st.session_state.total_characters = 0
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

# Função para criar um novo chat
def new_chat():
    current_date = datetime.now().strftime("%d/%m/%Y")
    chat_id = f"chat_{len(st.session_state.chats) + 1}"
    st.session_state.chats[chat_id] = {'date': current_date, 'messages': [], 'title': "Novo Chat"}
    st.session_state.current_chat_id = chat_id
    logger.info(f"Novo chat criado: {chat_id}")

# Barra lateral
with st.sidebar:
    st.button("Novo Chat", on_click=new_chat)
    st.markdown("---")
    st.markdown("### Chats Anteriores")
    for chat_id, chat_data in st.session_state.chats.items():
        if st.button(f"{chat_data['title']} - {chat_data['date']}", key=chat_id):
            st.session_state.current_chat_id = chat_id
            logger.info(f"Usuário mudou para o chat: {chat_id}")

# Conteúdo principal
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    st.image("assets/LOGO SUCESSO EM VENDAS HORIZONTAL AZUL.png", width=300)

st.markdown("<div class='centered-header'><h1>| Consultor I.A. |</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='centered-header'><h3>| Especialista em Eletromóveis |</h3></div>", unsafe_allow_html=True)

# Botões de prompt predefinidos
col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Vender Produto"):
            st.session_state.user_input = ("Me ajude a vender uma (...), preciso de ideias práticas e ações "
                                           "aplicáveis para meu time vender esse produto, preciso que enfatize suas "
                                           "qualidades reais e diferenciais e busque argumentos concisos que "
                                           "naturalmente me ajudem com possíveis objeções.")
    with col_b:
        if st.button("Criar Treinamento"):
            st.session_state.user_input = ("Me ajude a criar um treinamento de (...) com ferramentas e uma lógica de "
                                           "apresentação. Destrinche os tópicos com conteúdos mais práticos e aplicáveis.")
    with col_c:
        if st.button("Estratégia de Marketing"):
            st.session_state.user_input = ("Preciso de uma estratégia de marketing para aumentar a visibilidade e "
                                           "engajamento do nosso produto. Inclua ideias inovadoras que possam ser "
                                           "implementadas rapidamente e que aproveitem as tendências atuais do mercado.")

# Formulário de entrada
with st.form(key='input_form', clear_on_submit=True):
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        user_input = st.text_input(label='Digite sua mensagem', key='user_input')
        submit_button = st.form_submit_button(label="Enviar")

if submit_button:
    st.session_state.user_interactions += 1
    logger.info(f"Total de interações do usuário: {st.session_state.user_interactions}")
    
    st.session_state.chats[st.session_state.current_chat_id]['messages'].append(('user', user_input))
    
    if st.session_state.chats[st.session_state.current_chat_id]['title'] == "Novo Chat":
        st.session_state.chats[st.session_state.current_chat_id]['title'] = extract_title(user_input)
    
    with st.spinner("Gerando resposta..."):
        response = generate_response(user_input, context)
        
        typing_container = st.empty()
        display_typing_response(response, typing_container)
        
        typing_container.empty()
    
    st.session_state.chats[st.session_state.current_chat_id]['messages'].append(('agent', response))
    
    interaction_tokens = num_tokens_from_string(user_input) + num_tokens_from_string(response)
    interaction_chars = count_characters(user_input) + count_characters(response)
    st.session_state.total_tokens += interaction_tokens
    st.session_state.total_characters += interaction_chars
    logger.info(f"Tokens nesta interação: {interaction_tokens}")
    logger.info(f"Caracteres nesta interação: {interaction_chars}")
    logger.info(f"Total de tokens acumulados: {st.session_state.total_tokens}")
    logger.info(f"Total de caracteres acumulados: {st.session_state.total_characters}")

    cache_key = hashlib.md5((user_input + context[:100]).encode()).hexdigest()
    if cache_key in st.session_state.response_cache:
        logger.info("Esta resposta foi recuperada do cache.")

# Exibir histórico do chat atual
chat_history = st.session_state.chats[st.session_state.current_chat_id]['messages']
with st.container():
    st.write("Histórico:")
    chat_container = st.container()
    with chat_container:
        for role, message in chat_history:
            if role == 'user':
                st.markdown(f"<div class='user-message'>Usuário: {message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='agent-message'>Agente: {message}</div>", unsafe_allow_html=True)

# Script para tornar a barra lateral responsiva em dispositivos móveis
st.markdown("""
<script>
    var sidebar = document.querySelector('.sidebar');
    var sidebarToggle = document.createElement('button');
    sidebarToggle.textContent = '☰';
    sidebarToggle.className = 'sidebar-toggle';
    document.body.appendChild(sidebarToggle);

    sidebarToggle.addEventListener('click', function() {
        sidebar.classList.toggle('active');
    });
</script>
""", unsafe_allow_html=True)
