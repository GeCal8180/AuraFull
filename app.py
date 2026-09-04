import gradio as gr
import pandas as pd
import os
import docx
from docx.shared import Inches
import pdfplumber
import shutil
import re
import zipfile
import time
import base64
import json
import urllib.parse
import urllib.request
from datetime import datetime
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from duckduckgo_search import DDGS
from gradio_client import Client
from huggingface_hub import InferenceClient

# ==========================================
# 1. CHAVES E CONEXÕES
# ==========================================
chave_groq = os.environ.get("GROQ_API_KEY")
chave_hf = os.environ.get("HF_TOKEN")

cliente_groq = Groq(api_key=chave_groq)
cliente_hf = InferenceClient(token=chave_hf)
MODELO_GROQ = "llama-3.3-70b-versatile"
MODELO_VISAO = "llama-3.2-90b-vision-preview"

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=chave_hf, 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# 2. BANCO DE DADOS E DIRETÓRIOS
# ==========================================
DIRETORIO = "./Central_IA_Master"
DIR_CHATS = f"{DIRETORIO}/Historico_Chats"
DIR_MIDIA = f"{DIRETORIO}/Midia_Criada"
DIR_CASOS = f"{DIRETORIO}/Projetos_Salvos"
DIR_CHROMA = f"{DIRETORIO}/Banco_de_Dados_Vetorial"

for d in [DIRETORIO, DIR_CHATS, DIR_MIDIA, DIR_CASOS, DIR_CHROMA]:
    os.makedirs(d, exist_ok=True)

# --- FUNÇÕES DE APOIO ---
def listar_sessoes_chat():
    sessoes = [f.replace('.json', '') for f in os.listdir(DIR_CHATS) if f.endswith('.json')]
    sessoes.sort(reverse=True)
    return sessoes if sessoes else ["Sem histórico"]

def carregar_sessao_chat(id_sessao):
    if not id_sessao or id_sessao == "Sem histórico": return [], id_sessao
    try:
        with open(f"{DIR_CHATS}/{id_sessao}.json", "r", encoding="utf-8") as f: return json.load(f), id_sessao
    except: return [], id_sessao

def iniciar_novo_chat():
    novo_id = f"Chat_{datetime.now().strftime('%d%m_%H%M%S')}"
    return [], novo_id, gr.update(choices=listar_sessoes_chat(), value=novo_id)

def atualizar_galeria_imagens():
    imgs = []
    for d in [DIR_CASOS, DIR_MIDIA]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')): imgs.append(os.path.join(root, f))
    imgs.sort(key=os.path.getmtime, reverse=True)
    return imgs

def listar_arquivos_mortos():
    arquivos = []
    for d in [DIR_CASOS, DIR_MIDIA]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if not f.startswith('.'): arquivos.append(os.path.join(root, f))
    arquivos.sort(key=os.path.getmtime, reverse=True)
    return arquivos

# ==========================================
# 3. MOTORES DE GERAÇÃO
# ==========================================
def motor_gerar_imagem(prompt_desc, proporcao):
    try:
        w, h = (1080, 1920) if "Vertical" in proporcao else (1920, 1080) if "Horizontal" in proporcao else (1024, 1024)
        prompt_url = urllib.parse.quote(prompt_desc.strip())
        url = f"https://image.pollinations.ai/prompt/{prompt_url}?width={w}&height={h}&nologo=true&seed={int(time.time())}"
        caminho = f"{DIR_MIDIA}/Img_{datetime.now().strftime('%H%M%S')}.jpg"
        urllib.request.urlretrieve(url, caminho)
        return caminho
    except: return None

def motor_editar_imagem(imagem, prompt):
    try:
        res = cliente_hf.image_to_image(image=imagem, prompt=prompt, model="timbrooks/instruct-pix2pix")
        cam = f"{DIR_MIDIA}/Edit_{datetime.now().strftime('%H%M%S')}.jpg"
        res.save(cam)
        return cam
    except: return None

def motor_gerar_audio(texto):
    try:
        cam_audio = f"{DIR_MIDIA}/Voz_{datetime.now().strftime('%H%M%S')}.mp3"
        with open("temp.txt", "w", encoding="utf-8") as f: f.write(texto[:2500])
        os.system(f'edge-tts --voice pt-BR-AntonioNeural -f temp.txt --write-media "{cam_audio}"')
        return cam_audio
    except: return None

def motor_gerar_video(prompt, img=None):
    for t in range(3):
        try:
            if img:
                return Client("multimodalart/stable-video-diffusion", hf_token=chave_hf).predict(img, api_name="/video")
            return Client("multimodalart/zeroscope-v2", hf_token=chave_hf).predict(prompt[:150], api_name="/infer")
        except: time.sleep(3); continue
    return None

# ==========================================
# 4. CÉREBRO DO CHAT (CÓDIGO DE OURO)
# ==========================================
def responder_chat_ouro(mensagem, historico, persona, internet, id_sessao):
    texto_user = mensagem.get("text", "") if isinstance(mensagem, dict) else str(mensagem)
    arquivos = mensagem.get("files", []) if isinstance(mensagem, dict) else []
    contexto, imagens = "", []
    
    yield "⏳ *Código de Ouro processando...*"
    
    for f in arquivos:
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): imagens.append(f)
        else: 
            # Função simplificada de extração
            contexto += f"\n[DOC]: {f}\n"

    if internet and texto_user:
        try:
            res = DDGS().text(texto_user, max_results=3)
            contexto += "\n[WEB]: " + " ".join([r['body'] for r in res])
        except: pass

    sys_prompt = f"Você é o {persona} da IA 'Código de Ouro'. Responda com luxo, clareza e precisão. PODERES: Imagem [AÇÃO_IMAGEM: prompt | vertical], Edição [AÇÃO_EDITAR_IMAGEM: prompt], Áudio [AÇÃO_AUDIO: texto], Vídeo [AÇÃO_VIDEO: prompt], Gráficos ```mermaid."
    mensagens = [{"role": "system", "content": sys_prompt}]
    
    for h in (historico or []):
        if isinstance(h, dict): mensagens.append(h)
        else: 
            mensagens.append({"role": "user", "content": h[0]})
            mensagens.append({"role": "assistant", "content": h[1]})

    prompt_final = texto_user + contexto
    if imagens:
        with open(imagens[-1], "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
        mensagens.append({"role": "user", "content": [{"type": "text", "text": prompt_final}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]})
        modelo = MODELO_VISAO
    else:
        mensagens.append({"role": "user", "content": prompt_final})
        modelo = MODELO_GROQ

    stream = cliente_groq.chat.completions.create(messages=mensagens, model=modelo, stream=True)
    full_res = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            full_res += chunk.choices[0].delta.content
            yield re.sub(r'\[AÇÃO_.*?:.*?\]', '⚙️ *Acionando motor de mídia...*', full_res)

    # Processar Gatilhos (Mesma lógica das versões anteriores, mas com nomes limpos)
    res_final = full_res
    # ... (lógica de gatilhos de mídia omitida aqui para brevidade, mas mantida no app.py final)
    yield res_final

# ==========================================
# 5. DESIGN SYSTEM LUXURY (O HACK DO LOGIN)
# ==========================================

# ESTE BLOCO DE CSS É O QUE FAZ A MÁGICA NA TELA DE LOGIN E NO SITE
CSS_DOLA_GOLD = """
/* RESET E FUNDO */
footer {display: none !important;}
body, .gradio-container { background-color: #050505 !important; font-family: 'Inter', sans-serif !important; }

/* LOGO NO LOGIN */
#auth-header { text-align: center; margin-bottom: 2rem; }
.logo-gold { 
    color: #D4AF37; 
    font-size: 3rem; 
    font-weight: 900; 
    letter-spacing: 5px; 
    text-shadow: 0 0 30px rgba(212, 175, 55, 0.4);
    margin-bottom: 0.5rem;
}

/* TELA DE LOGIN (O HACK) */
.wrap.gradio-container { justify-content: center !important; }
.auth-box { 
    background: #0D0D0D !important; 
    border: 1px solid #D4AF37 !important; 
    border-radius: 40px !important; 
    padding: 3rem !important; 
    box-shadow: 0 20px 50px rgba(0,0,0,0.9) !important;
}
#login-btn { 
    background: linear-gradient(135deg, #D4AF37 0%, #B58500 100%) !important; 
    color: #000 !important; 
    border-radius: 50px !important; 
    font-weight: 900 !important; 
    text-transform: uppercase !important;
    border: none !important;
    padding: 1rem !important;
    transition: 0.3s !important;
}
#login-btn:hover { transform: scale(1.03); box-shadow: 0 0 20px rgba(212, 175, 55, 0.5); }

/* LAYOUT DOLA NO SITE PRINCIPAL */
.gradio-container { max-width: 900px !important; }
.tab-nav { border: none !important; justify-content: center !important; gap: 10px !important; }
.tab-nav button { 
    border-radius: 50px !important; 
    background: #111 !important; 
    border: 1px solid #222 !important; 
    color: #888 !important; 
    padding: 8px 25px !important; 
}
.tab-nav button.selected { 
    background: #D4AF37 !important; 
    color: #000 !important; 
    font-weight: 700 !important;
}

/* BALÕES DE CHAT */
.message.user { border: 1px solid #D4AF37 !important; border-radius: 25px 25px 5px 25px !important; background: rgba(212,175,55,0.05) !important; }
.message.bot { border: 1px solid #222 !important; border-radius: 25px 25px 25px 5px !important; background: #111 !important; }

/* INPUTS DARK */
input, textarea { background: #0A0A0A !important; border: 1px solid #333 !important; border-radius: 20px !important; color: gold !important; }
"""

# HTML DA LOGO
LOGO_HTML = """
<div id="auth-header">
    <h1 class="logo-gold">CÓDIGO DE OURO</h1>
    <p style="color: #666; letter-spacing: 3px; font-size: 0.8rem;">SISTEMA DE ALTA PERFORMANCE</p>
</div>
"""

# ==========================================
# 6. INTERFACE FINAL
# ==========================================
with gr.Blocks(title="Código de Ouro", css=CSS_DOLA_GOLD, theme=gr.themes.Soft()) as interface:
    
    gr.HTML(LOGO_HTML) # Logo no topo do site
    
    id_sessao_atual = gr.State(f"Chat_{datetime.now().strftime('%d%m_%H%M%S')}")

    with gr.Sidebar(open=False):
        gr.Markdown("### 📜 Histórico")
        btn_novo = gr.Button("➕ Novo Chat", variant="primary")
        lista_chats = gr.Dropdown(choices=listar_sessoes_chat(), label="Conversas Salvas")
        btn_load = gr.Button("Abrir Conversa")
        gr.Button("🔄 Atualizar Lista").click(lambda: gr.update(choices=listar_sessoes_chat()), None, lista_chats)

    with gr.Tabs():
        with gr.TabItem("💬 Chat IA"):
            with gr.Row():
                persona = gr.Dropdown(choices=["Assistente Padrão", "Especialista em Marketing", "Analista de Dados"], value="Assistente Padrão", label="Especialidade", scale=2)
                net = gr.Checkbox(label="🌐 Web", scale=1)
            
            chat = gr.ChatInterface(
                fn=responder_chat_ouro, multimodal=True, additional_inputs=[persona, net, id_sessao_atual],
                chatbot=gr.Chatbot(height=600, show_label=False),
                textbox=gr.MultimodalTextbox(placeholder="Envie uma mensagem ou arquivo...", container=False)
            )
            
            btn_load.click(carregar_sessao_chat, lista_chats, [chat.chatbot, id_sessao_atual])
            btn_novo.click(iniciar_novo_chat, None, [chat.chatbot, id_sessao_atual, lista_chats])

        with gr.TabItem("📑 Documentos"):
            with gr.Row():
                files = gr.File(label="Upload (PDF/Excel)", file_count="multiple")
                inst = gr.Textbox(label="Instrução", placeholder="O que devo analisar?")
            btn_doc = gr.Button("Iniciar Análise Profissional", variant="primary")
            res_doc = gr.Textbox(label="Resultado", lines=10)

        with gr.TabItem("🗂️ Galeria"):
            btn_att_gal = gr.Button("🔄 Sincronizar Galeria Visual", variant="primary")
            gal = gr.Gallery(columns=4, height="auto")
            btn_att_gal.click(atualizar_galeria_imagens, None, gal)

# ==========================================
# 7. LANÇAMENTO COM AUTH REESTILIZADO
# ==========================================
usuarios = []
for i in ["", "_1", "_2", "_3"]:
    u, s = os.environ.get(f"LOGIN_USUARIO{i}" if i=="" else f"USUARIO{i}"), os.environ.get(f"LOGIN_SENHA{i}" if i=="" else f"SENHA{i}")
    if u and s: usuarios.append((u, s))

interface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 10000)),
    auth=usuarios,
    auth_message=LOGO_HTML, # Injeta a logo na tela de login
    js="() => { document.body.classList.add('dark'); }" # Força Dark Mode no Login
)
