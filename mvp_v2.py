import streamlit as st

st.set_page_config(page_title='Consultor da Sucesso em Vendas', layout="wide")

# Resto das importações
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
import docx
import PyPDF2
import os
from google.auth import load_credentials_from_file
from langchain.globals import set_verbose
import tornado.websocket
import time
from datetime import datetime
import logging
import tiktoken
import hashlib
import re
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import google.auth
from google.auth.exceptions import DefaultCredentialsError

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Definir a verbosidade
set_verbose(True)

# Carregar variáveis de ambiente
load_dotenv()

# Resto do seu código...
try:
    # Criar credenciais a partir dos secrets
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )


except Exception as e:
    logger.error(f"Erro ao carregar credenciais: {e}")
    st.error(f"Erro ao carregar credenciais: {str(e)}. Por favor, verifique a configuração.")
    st.stop()

# Inicializar o modelo Gemini com as credenciais carregadas
try:
    # Inicializar o modelo Gemini com as credenciais
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.3,
        credentials=credentials
    )
    logger.info("Modelo Gemini inicializado com sucesso")
except Exception as e:
    logger.error(f"Erro ao inicializar o modelo Gemini: {e}")
    st.error(f"Erro ao inicializar o modelo de IA: {str(e)}. Por favor, tente novamente mais tarde.")
    st.stop()

# Função para contar tokens
def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Função para contar caracteres
def count_characters(text):
    return len(text)

# Função para carregar e processar arquivos JSON
def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {filepath}")
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao decodificar o JSON no arquivo {filepath}: {e}")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o arquivo JSON {filepath}: {e}")

# Função para carregar e processar arquivos DOCX
def load_docx(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo DOCX não encontrado: {filepath}")
    try:
        doc = docx.Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o arquivo DOCX {filepath}: {e}")

# Função para carregar e processar arquivos PDF
def load_pdf(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {filepath}")
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() if page.extract_text() else ""
        return text
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o arquivo PDF {filepath}: {e}")

# Função para carregar todos os arquivos na pasta materiais
def load_materials(directory='materiais'):
    materials = []
    total_tokens = 0
    total_chars = 0
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Pasta de materiais não encontrada: {directory}")
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.json'):
            try:
                content = load_json(filepath)
                materials.append(content)
                content_str = str(content)
                total_tokens += num_tokens_from_string(content_str)
                total_chars += count_characters(content_str)
            except Exception as e:
                logger.error(f"Erro ao carregar arquivo JSON: {e}")
        elif filename.endswith('.docx'):
            try:
                content = load_docx(filepath)
                materials.append(content)
                total_tokens += num_tokens_from_string(content)
                total_chars += count_characters(content)
            except Exception as e:
                logger.error(f"Erro ao carregar arquivo DOCX: {e}")
        elif filename.endswith('.pdf'):
            try:
                content = load_pdf(filepath)
                materials.append(content)
                total_tokens += num_tokens_from_string(content)
                total_chars += count_characters(content)
            except Exception as e:
                logger.error(f"Erro ao carregar arquivo PDF: {e}")

    materials_text = "\n\n".join(map(str, materials))
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
    # Gerar uma chave única para o cache
    cache_key = hashlib.md5((user_input + context[:100]).encode()).hexdigest()
    # Verificar se a resposta está no cache
    if cache_key in st.session_state.response_cache:
        logger.info("Resposta encontrada no cache")
        cached_response = st.session_state.response_cache[cache_key]
        return cached_response

    prompt = f"{context}\n\nUsuário: {user_input}\nChatbot:"
    input_tokens = num_tokens_from_string(prompt)
    input_chars = count_characters(prompt)
    logger.info(f"Tokens na entrada: {input_tokens}")
    logger.info(f"Caracteres na entrada: {input_chars}")

    model = ChatPromptTemplate.from_template(prompt) | llm
    try:
        logger.info("Iniciando geração de resposta")
        response = model.invoke({'input': prompt})
        logger.info("Resposta gerada com sucesso")
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        response_tokens = num_tokens_from_string(response_content)
        response_chars = count_characters(response_content)
        logger.info(f"Tokens na resposta: {response_tokens}")
        logger.info(f"Caracteres na resposta: {response_chars}")
        
        total_tokens = input_tokens + response_tokens
        total_chars = input_chars + response_chars
        logger.info(f"Total de tokens nesta interação: {total_tokens}")
        logger.info(f"Total de caracteres nesta interação: {total_chars}")
        
        # Armazenar a resposta no cache
        st.session_state.response_cache[cache_key] = response_content
        
        return response_content
    except Exception as e:
        logger.error(f"Erro detalhado ao gerar resposta: {str(e)}")
        return f"Ocorreu um erro ao gerar a resposta: {str(e)}. Por favor, tente novamente."

# Função para exibir a resposta gradualmente como se estivesse digitando
def display_typing_response(response_text, container):
    typing_speed = 0.01  # Velocidade de digitação (em segundos por caractere)
    typed_text = ""
    for char in response_text:
        typed_text += char
        container.markdown(f"<div class='agent-message'>{typed_text}</div>", unsafe_allow_html=True)
        time.sleep(typing_speed)

# Função para extrair título do chat
def extract_title(message):
    # Extrair as primeiras duas ou três palavras significativas
    words = re.findall(r'\b\w+\b', message)
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    return "Novo Chat"

# Adicionando estilo personalizado para tema claro
st.markdown(
"""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
body, .stApp {
    background-color: #FFFFFF;
    color: #000000;
}

@import url('https://fonts.googleapis.com/css2?family=Sofia+Pro:wght@400&display=swap');

* {
    font-family: 'Sofia Pro', sans-serif;
    color: #000000;
}

.stImage > img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    font-family: 'Sofia Pro', sans-serif;
    color: #000000;
    text-align: center;
    font-weight: bold;
}

.centered-header {
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    margin-bottom: 20px;
}

.stTextInput div label {
    font-family: 'Sofia Pro', sans-serif;
    color: #000000;
}

.stTextInput div input {
    font-family: 'Sofia Pro', sans-serif;
    color: #000000;
    background-color: #F0F0F0;
    max-width: 100%;
    margin: 0 auto;
}

.stButton button {
    background-color: #E0E0E0;
    color: #000000;
    font-weight: bold;
    font-family: 'Sofia Pro', sans-serif;
    border-radius: 5px;
    padding: 10px;
    border: none;
}

.user-message {
    background-color: #F0F0F0;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}

.agent-message {
    background-color: #E8E8E8;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}

.scroll-container {
    max-height: 80vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;
}

.chat-input {
    margin-top: 20px;
    display: flex;
    justify-content: center;
}

.sidebar .stButton button {
    width: 100%;
    margin-bottom: 10px;
    background-color: #E0E0E0;
    color: #000000;
    font-weight: bold;
    font-family: 'Sofia Pro', sans-serif;
    border-radius: 5px;
    padding: 10px;
    border: none;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
}

.stForm div {
    margin: 0;  /* Remove margens ao redor do formulário */
    padding: 0;  /* Remove preenchimento ao redor do formulário */
}

/* Estilo para dispositivos móveis */
@media (max-width: 768px) {
    .sidebar {
        position: fixed;
        top: 0;
        left: -250px;
        height: 100vh;
        width: 250px;
        background-color: #FFFFFF;
        transition: 0.3s;
        z-index: 1000;
    }
    .stImage > img {
        display: block;
        margin-left: calc(1cm + auto);
        margin-right: auto;
    }
    .sidebar.active {
        left: 0;
    }

    .stTextInput div input {
        max-width: 95%;
    }
}
</style>
""",
unsafe_allow_html=True
)

# Carregar materiais e definir contexto com indicador de carregamento
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

# Inicializar o estado da sessão para múltiplos chats
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

# Centralizar o header
st.markdown("<div class='centered-header'><h1>| Consultor I.A. |</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='centered-header'><h3>| Especialista em Eletromóveis |</h3></div>", unsafe_allow_html=True)

# Adicionando botões de prompt predefinidos
col1, col2, col3 = st.columns([1, 5, 1])  # Ajuste para alinhar com o campo de entrada
with col2:
    col_a, col_b, col_c = st.columns(3)  # Dividir a coluna central em três para os botões
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
        if st.button("Comparativo de produto"):
            st.session_state.user_input = ("Quero construir uma abordagem de vendas para uma (inserir modelo do produto) "
                                           "considerando todas as etapas, desde a abordagem até o fechamento e considerando "
                                           "os principais diferenciais do produto. Foque nas perguntas de pesquisa e crie um "
                                           "caderno de objeções, contornando as principais com relação a produtos similares "
                                           "do (produto a ser comparado).")

# Inicializar o estado da sessão para a entrada do usuário
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# Modificação na parte do formulário de entrada
with st.form(key='input_form', clear_on_submit=True):
    col1, col2, col3 = st.columns([1, 5, 1])  # Ajustado para dar mais espaço à coluna central
    with col2:
        # Capture a entrada do usuário
        user_input = st.text_input(label='Digite sua mensagem', key='user_input')
        submit_button = st.form_submit_button(label="Enviar")

if submit_button:
    st.session_state.user_interactions += 1
    logger.info(f"Total de interações do usuário: {st.session_state.user_interactions}")
    # Adicionar mensagem do usuário ao histórico
    st.session_state.chats[st.session_state.current_chat_id]['messages'].append(('user', user_input))

    # Atualizar o título do chat com base na nova entrada
    if st.session_state.chats[st.session_state.current_chat_id]['title'] == "Novo Chat":
        st.session_state.chats[st.session_state.current_chat_id]['title'] = extract_title(user_input)

    # Gerar resposta
    with st.spinner("Gerando resposta..."):
        response = generate_response(user_input, context)
        
        # Exibir resposta gradualmente
        typing_container = st.empty()
        display_typing_response(response, typing_container)
        
        # Após exibir, remover a resposta da visualização direta
        typing_container.empty()

    # Adicionar resposta ao histórico
    st.session_state.chats[st.session_state.current_chat_id]['messages'].append(('agent', response))

    # Atualizar o contador de tokens e caracteres total
    interaction_tokens = num_tokens_from_string(user_input) + num_tokens_from_string(response)
    interaction_chars = count_characters(user_input) + count_characters(response)
    st.session_state.total_tokens += interaction_tokens
    st.session_state.total_characters += interaction_chars
    logger.info(f"Tokens nesta interação: {interaction_tokens}")
    logger.info(f"Caracteres nesta interação: {interaction_chars}")
    logger.info(f"Total de tokens acumulados: {st.session_state.total_tokens}")
    logger.info(f"Total de caracteres acumulados: {st.session_state.total_characters}")

    # Log de informação sobre o uso do cache
    cache_key = hashlib.md5((user_input + context[:100]).encode()).hexdigest()
    if cache_key in st.session_state.response_cache:
        logger.info("Esta resposta foi recuperada do cache.")

# Exibir histórico do chat atual
chat_history = st.session_state.chats[st.session_state.current_chat_id]['messages']
with st.container():
    st.write("Histórico:")
    chat_container = st.container()
    with chat_container:
        # Exibir pergunta antes da resposta
        for i in range(0, len(chat_history), 2):
            user_message = chat_history[i]
            agent_message = chat_history[i+1] if i+1 < len(chat_history) else ('agent', '')
            st.markdown(f"<div class='agent-message'>Agente: {agent_message[1]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='user-message'>Usuário: {user_message[1]}</div>", unsafe_allow_html=True)

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

if __name__ == "__main__":
    logger.info("Aplicação iniciada")
