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
# 1. CHAVES MESTRES E CONEXÕES
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
# 2. DIRETÓRIOS E BANCO DE DADOS LOCAL
# ==========================================
DIRETORIO = "./Central_IA_Master"
DIR_CHROMA = f"{DIRETORIO}/Banco_de_Dados_Vetorial"
DIR_CASOS = f"{DIRETORIO}/Projetos_Salvos"
DIR_MIDIA = f"{DIRETORIO}/Midia_Criada"
DIR_CHATS = f"{DIRETORIO}/Historico_Chats"

for d in [DIRETORIO, DIR_CASOS, DIR_MIDIA, DIR_CHATS]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 2.5. CAPTURA DA LOGOMARCA OFICIAL
# ==========================================
TAG_LOGO = ""
nome_logo = "chamariz-sem-fundo.png"

if os.path.exists(nome_logo):
    with open(nome_logo, "rb") as f:
        b64_logo = base64.b64encode(f.read()).decode('utf-8')
        TAG_LOGO = f'<img src="data:image/png;base64,{b64_logo}" style="max-height: 85px; margin: 0 auto 15px auto; display: block; filter: drop-shadow(0px 4px 15px rgba(212, 175, 55, 0.4));" alt="Código de Ouro" />'

# --- GESTÃO DE SESSÕES ---
def listar_sessoes_chat():
    sessoes = [f.replace('.json', '') for f in os.listdir(DIR_CHATS) if f.endswith('.json')]
    sessoes.sort(reverse=True)
    return sessoes if sessoes else ["Nenhuma conversa salva"]

def carregar_sessao_chat(id_sessao):
    if not id_sessao or id_sessao == "Nenhuma conversa salva": return [], id_sessao
    try:
        with open(f"{DIR_CHATS}/{id_sessao}.json", "r", encoding="utf-8") as f: return json.load(f), id_sessao
    except: return [], id_sessao

def iniciar_novo_chat():
    novo_id = f"Chat_{datetime.now().strftime('%d%m_%H%M%S')}"
    return [], novo_id, gr.update(choices=listar_sessoes_chat(), value=novo_id)

def listar_arquivos_mortos():
    arquivos = []
    for d in [DIR_CASOS, DIR_MIDIA]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(('.docx', '.pdf', '.mp3', '.jpg', '.mp4')): arquivos.append(os.path.join(root, f))
    arquivos.sort(key=os.path.getmtime, reverse=True)
    return arquivos

def atualizar_galeria_imagens():
    imgs = []
    for d in [DIR_CASOS, DIR_MIDIA]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(('.jpg', '.png', '.jpeg', '.webp')): imgs.append(os.path.join(root, f))
    imgs.sort(key=os.path.getmtime, reverse=True)
    return imgs

# ==========================================
# 3. EXTRAÇÃO MULTIMÍDIA E DADOS
# ==========================================
def encode_file_b64(caminho):
    with open(caminho, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

def extrair_texto(arquivo):
    caminho = arquivo.name if hasattr(arquivo, 'name') else arquivo
    texto = ""
    try:
        if caminho.lower().endswith('.pdf'):
            with pdfplumber.open(caminho) as pdf:
                for p in pdf.pages:
                    txt = p.extract_text()
                    if txt: texto += txt + "\n"
        elif caminho.lower().endswith(('.xlsx', '.csv')): texto = (pd.read_excel(caminho) if caminho.lower().endswith('.xlsx') else pd.read_csv(caminho)).to_string() 
        elif caminho.lower().endswith('.docx'):
            for p in docx.Document(caminho).paragraphs: texto += p.text + "\n"
        elif caminho.lower().endswith(('.mp3', '.ogg', '.wav')):
            with open(caminho, "rb") as file: texto = f"[TRANSCRIÇÃO]: {cliente_groq.audio.transcriptions.create(file=(caminho, file.read()), model='whisper-large-v3').text}\n"
        return texto
    except: return ""

# ==========================================
# 4. MOTORES DE MÍDIA
# ==========================================
def motor_gerar_imagem(prompt_desc, proporcao="Vertical (TikTok/Reels)"):
    try:
        w, h = (1080, 1920) if "vertical" in proporcao.lower() else (1920, 1080) if "horizontal" in proporcao.lower() else (1024, 1024)
        prompt_clean = urllib.parse.quote(prompt_desc.strip())
        url = f"https://image.pollinations.ai/prompt/{prompt_clean}?width={w}&height={h}&nologo=true&seed={int(time.time())}"
        nome_arq = f"{DIR_MIDIA}/Img_{datetime.now().strftime('%H%M%S')}.jpg"
        urllib.request.urlretrieve(url, nome_arq)
        return nome_arq
    except: return None

def motor_editar_imagem(caminho_imagem, prompt_edicao):
    for tentativa in range(3):
        try:
            res = cliente_hf.image_to_image(image=caminho_imagem, prompt=prompt_edicao, model="timbrooks/instruct-pix2pix")
            caminho_saida = f"{DIR_MIDIA}/Edit_{datetime.now().strftime('%H%M%S')}.jpg"
            res.save(caminho_saida)
            return caminho_saida
        except: time.sleep(2); continue
    return None

def motor_gerar_audio(texto):
    try:
        cam_txt = f"{DIR_MIDIA}/temp_{datetime.now().strftime('%H%M%S')}.txt"
        cam_audio = f"{DIR_MIDIA}/Voz_{datetime.now().strftime('%H%M%S')}.mp3"
        with open(cam_txt, "w", encoding="utf-8") as f: f.write(texto[:2500].replace('*', ''))
        os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{cam_txt}" --write-media "{cam_audio}"')
        if os.path.exists(cam_txt): os.remove(cam_txt)
        return cam_audio
    except: return None

def motor_gerar_video(prompt_cena, imagem_base=None):
    for tentativa in range(4):
        try:
            if imagem_base: return Client("multimodalart/stable-video-diffusion", hf_token=chave_hf).predict(imagem_base, api_name="/video")
            else: return Client("multimodalart/zeroscope-v2", hf_token=chave_hf).predict(prompt_cena[:150], api_name="/infer")
        except: time.sleep(3); continue
    return None

# ==========================================
# 5. O CHAT AGÊNTICO
# ==========================================
def responder_chat_central(mensagem, historico, persona, usar_internet, id_sessao):
    texto_usuario = mensagem.get("text", "") if isinstance(mensagem, dict) else str(mensagem)
    arquivos = mensagem.get("files", []) if isinstance(mensagem, dict) else []
    
    contexto_extra = ""
    imagens_anexadas = []
    
    yield "⏳ *Sincronizando com a base de dados...*"
    
    for arq in arquivos:
        ext = arq.lower()
        if ext.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            imagens_anexadas.append(arq)
            yield f"👁️ *Analisando a imagem...*"
        else:
            yield f"📄 *Extraindo dados do documento...*"
            contexto_extra += f"\n[DOCUMENTO]:\n{extrair_texto(arq)}\n"
            
    if usar_internet and texto_usuario:
        yield "🌐 *Pesquisando na internet...*"
        try:
            resultados = DDGS().text(texto_usuario, max_results=4)
            contexto_extra += "\n\n[DADOS WEB]:\n" + "\n".join([f"{r['title']} - {r['body']}" for r in resultados])
        except: pass

    yield "🧠 *Processando...*"

    sys_prompt = f"""Você é a IA "Código de Ouro" operando no perfil {persona}.
Aja de forma inteligente, direta e com alto nível de execução.

PODERES EXECUTIVOS NO CHAT:
1. GERAR IMAGEM: [AÇÃO_IMAGEM: prompt em inglês detalhado 8k photorealistic | vertical]
2. EDITAR IMAGEM: [AÇÃO_EDITAR_IMAGEM: comando em inglês. Ex: add sunglasses]
3. GERAR VÍDEO: [AÇÃO_VIDEO: prompt curto em inglês]
4. GERAR ÁUDIO: [AÇÃO_AUDIO: texto para falar em português]
5. GRÁFICOS: Use blocos ```mermaid para visualização de dados numéricos."""

    mensagens = [{"role": "system", "content": sys_prompt}]
    
    if historico:
        for item in historico:
            if isinstance(item, dict):
                mensagens.append({"role": item.get("role", "user"), "content": str(item.get("content", ""))})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                if item[0]: mensagens.append({"role": "user", "content": str(item[0])})
                if item[1]: mensagens.append({"role": "assistant", "content": str(item[1])})

    texto_final = (texto_usuario + contexto_extra).strip()

    if imagens_anexadas:
        conteudo_multimodal = [{"type": "text", "text": texto_final if texto_final else "Analise esta imagem."}]
        for img in imagens_anexadas:
            conteudo_multimodal.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_file_b64(img)}"}})
        mensagens.append({"role": "user", "content": conteudo_multimodal})
        modelo_escolhido = MODELO_VISAO
    else:
        mensagens.append({"role": "user", "content": texto_final if texto_final else "Olá!"})
        modelo_escolhido = MODELO_GROQ

    stream = cliente_groq.chat.completions.create(messages=mensagens, model=modelo_escolhido, max_tokens=4000, stream=True)
    
    resposta_acumulada = ""
    for pedaco in stream:
        delta = pedaco.choices[0].delta.content
        if delta:
            resposta_acumulada += delta
            yield re.sub(r'\[AÇÃO_\w+:.*?\]', '⚙️ *(Acionando motor multimídia...)*', resposta_acumulada)

    # Processamento de Gatilhos
    anexos_html = ""
    match_img = re.search(r'\[AÇÃO_IMAGEM:\s*(.*?)(?:\|\s*(\w+))?\]', resposta_acumulada)
    if match_img:
        prompt_i = match_img.group(1).strip()
        prop_i = "Vertical" if (match_img.group(2) and "vertical" in match_img.group(2).lower()) else "Quadrado"
        cam_gerada = motor_gerar_imagem(prompt_i, prop_i)
        if cam_gerada:
            b64_img = encode_file_b64(cam_gerada)
            anexos_html += f"\n\n**🖼️ Imagem Gerada:**\n<img src='data:image/jpeg;base64,{b64_img}' style='max-width:100%; height:auto; border-radius:15px; border: 1px solid rgba(212,175,55,0.4); margin-top:10px;' />\n"

    match_edit = re.search(r'\[AÇÃO_EDITAR_IMAGEM:\s*(.*?)\]', resposta_acumulada)
    if match_edit and imagens_anexadas:
        prompt_e = match_edit.group(1).strip()
        cam_edit = motor_editar_imagem(imagens_anexadas[-1], prompt_e)
        if cam_edit:
            b64_img = encode_file_b64(cam_edit)
            anexos_html += f"\n\n**✨ Imagem Editada:**\n<img src='data:image/jpeg;base64,{b64_img}' style='max-width:100%; height:auto; border-radius:15px; border: 1px solid rgba(212,175,55,0.4); margin-top:10px;' />\n"

    match_aud = re.search(r'\[AÇÃO_AUDIO:\s*(.*?)\]', resposta_acumulada)
    if match_aud:
        texto_loc = match_aud.group(1).strip()
        cam_aud = motor_gerar_audio(texto_loc)
        if cam_aud:
            b64_aud = encode_file_b64(cam_aud)
            anexos_html += f"\n\n**🔊 Locução:**\n<audio controls src='data:audio/mp3;base64,{b64_aud}' style='width:100%; margin-top:10px;'></audio>\n"

    match_vid = re.search(r'\[AÇÃO_VIDEO:\s*(.*?)\]', resposta_acumulada)
    if match_vid:
        prompt_v = match_vid.group(1).strip()
        img_referencia = imagens_anexadas[-1] if imagens_anexadas else None
        cam_vid = motor_gerar_video(prompt_v, img_referencia)
        if cam_vid:
            b64_vid = encode_file_b64(cam_vid)
            anexos_html += f"\n\n**🎥 Vídeo Gerado:**\n<video controls style='max-width:100%; border-radius:15px; border: 1px solid rgba(212,175,55,0.4); margin-top:10px;' src='data:video/mp4;base64,{b64_vid}'></video>\n"

    resposta_final_limpa = re.sub(r'\[AÇÃO_\w+:.*?\]', '', resposta_acumulada).strip() + anexos_html
    yield resposta_final_limpa

    # Auto-Save
    try:
        sessao_alvo = id_sessao if (id_sessao and id_sessao != "Nenhuma conversa salva") else f"Chat_{datetime.now().strftime('%d%m_%H%M%S')}"
        arq_sessao = f"{DIR_CHATS}/{sessao_alvo}.json"
        historico_atual = []
        if os.path.exists(arq_sessao):
            with open(arq_sessao, "r", encoding="utf-8") as f: historico_atual = json.load(f)
        historico_atual.append({"role": "user", "content": texto_usuario if texto_usuario else "[Arquivo/Mídia]"})
        historico_atual.append({"role": "assistant", "content": resposta_final_limpa})
        with open(arq_sessao, "w", encoding="utf-8") as f: json.dump(historico_atual, f, ensure_ascii=False, indent=4)
    except: pass

# ==========================================
# 6. FUNÇÕES DE ARQUIVO
# ==========================================
def exportar_conversa_docx(historico):
    if not historico: return None
    pasta = f"{DIR_CASOS}/Exportacoes_{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(pasta, exist_ok=True)
    cam_word = f"{pasta}/Historico_do_Chat.docx"
    doc = docx.Document()
    doc.add_heading('Histórico da Conversa', 0)
    for item in historico:
        if isinstance(item, dict):
            autor = "Você:" if item.get("role") == "user" else "Código de Ouro:"
            doc.add_heading(autor, level=2)
            doc.add_paragraph(re.sub(r'<.*?>', '', item.get("content", "")))
    doc.save(cam_word)
    return cam_word

def gerar_backup_zip():
    cam = "./Arquivos_Exportados.zip"
    with zipfile.ZipFile(cam, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(DIRETORIO):
            if "Banco_de_Dados_Vetorial" not in root:
                for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), DIRETORIO))
    return cam

def gerar_dossie_lote(arquivos, instrucao, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Forneça as instruções de análise.", None, "", ""
    palavras = 0
    try:
        progresso(0.1, desc="Lendo documentos...")
        pasta = f"{DIR_CASOS}/Analise_{datetime.now().strftime('%d_%m_%Y__%Hh%M')}"
        os.makedirs(pasta, exist_ok=True)
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for idx, arq in enumerate(arquivos):
                txt = extrair_texto(arq)
                palavras += len(txt.split())
                banco.add_texts([f"[FONTE: {os.path.basename(arq.name)}]\n{c}" for c in fatiador.split_text(txt)])
            
        progresso(0.5, desc="Analisando dados...")
        contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        prompt = f"Você é um Analista Especialista. DADOS:\n{contexto}\n\nINSTRUÇÃO: {instrucao}"
        resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ, max_tokens=4000).choices[0].message.content
        
        cam_word = f"{pasta}/Resultado.docx"
        doc = docx.Document()
        doc.add_heading('Análise Oficial', 0)
        doc.add_paragraph(resposta)
        doc.save(cam_word)
        return "✅ Análise Concluída!", cam_word, resposta, f"📊 Total lido: {palavras} palavras."
    except Exception as e: return f"Erro: {e}", None, "", ""

# ==========================================
# 7. DESIGN SYSTEM: WORKSTATION RESPONSIVA
# ==========================================

tema_ouro = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#000000", body_background_fill_dark="#000000",
    background_fill_primary="#000000", background_fill_primary_dark="#000000",
    background_fill_secondary="#050505", background_fill_secondary_dark="#050505",
    block_background_fill="#050505", block_background_fill_dark="#050505",
    border_color_primary="#1A1A1A", border_color_primary_dark="#1A1A1A",
    block_border_width="0px"
)

PWA_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="theme-color" content="#000000">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
"""

LOGIN_HACK = """
<style>
    body, main, .wrap { background-color: #000000 !important; color: #fff !important;}
    form { background: #0A0A0A !important; border: 1px solid rgba(212, 175, 55, 0.4) !important; border-radius: 30px !important; box-shadow: 0 15px 50px rgba(0,0,0,0.9) !important; padding: 40px !important; max-width: 90% !important; margin: auto !important; }
    button.primary { background: linear-gradient(145deg, #D4AF37, #AA7C11) !important; color: #000000 !important; font-weight: 800 !important; text-transform: uppercase !important; border-radius: 30px !important; border: none !important; font-size: 16px !important; letter-spacing: 1px !important; margin-top: 15px !important; transition: 0.3s !important;}
    button.primary:hover { transform: scale(1.02); box-shadow: 0 0 15px rgba(212, 175, 55, 0.5) !important; }
    input { background-color: #111111 !important; border: 1px solid #333 !important; border-radius: 20px !important; color: #D4AF37 !important; padding: 12px !important; }
    input:focus { border-color: #D4AF37 !important; outline: none !important; }
    form h2 { display: none !important; }
</style>
<div style="text-align: center; margin-bottom: 30px;">
    [LOGO_PLACEHOLDER]
    <h1 style="color: #D4AF37; font-size: clamp(24px, 6vw, 38px); font-weight: 900; margin: 0; letter-spacing: 2px; font-family: 'Inter', sans-serif; text-shadow: 0px 4px 25px rgba(212, 175, 55, 0.4);">CÓDIGO DE OURO</h1>
    <p style="color: #666; font-size: 12px; margin-top: 5px; font-weight: 700; letter-spacing: 3px;">ACESSO RESTRITO</p>
</div>
""".replace("[LOGO_PLACEHOLDER]", TAG_LOGO)

CSS_APP = """
/* PREVINE VAZAMENTO NO MOBILE E TRAVA X */
body, html { margin: 0 !important; padding: 0 !important; background-color: #000000 !important; overflow-x: hidden !important; max-width: 100vw !important;}
footer {display: none !important;}

/* CONTAINER PRINCIPAL E BOTÕES FIXOS */
.gradio-container { max-width: 100vw !important; border: none !important; overflow-x: hidden !important; box-sizing: border-box !important;}
.contain { padding: 0 !important; }

/* REVELA E ESTILIZA O BOTÃO DE ABRIR SIDEBAR */
button[aria-label="Open sidebar"], button[title="Open sidebar"], .sidebar-button {
    background-color: #0E0E0E !important;
    border: 1px solid #D4AF37 !important;
    border-radius: 50% !important;
    z-index: 99999 !important;
    box-shadow: 0 0 10px rgba(212, 175, 55, 0.3) !important;
    left: 15px !important; 
    top: 15px !important;
}
button[aria-label="Open sidebar"] svg, button[title="Open sidebar"] svg {
    color: #D4AF37 !important;
    stroke: #D4AF37 !important;
}

/* SIDEBAR NATIVA */
.sidebar { background-color: #050505 !important; border-right: 1px solid #111 !important; width: 280px !important; padding: 20px !important; }
.logo-container { text-align: center; padding: 10px 0 20px 0; border-bottom: 1px solid #111; margin-bottom: 20px;}
.logo-title { color: #D4AF37; font-size: 22px; font-weight: 900; margin: 0; letter-spacing: 2px; text-shadow: 0 0 15px rgba(212,175,55,0.2);}

/* BOTOES SECUNDARIOS DARK */
button.secondary, .dropdown { background-color: #111 !important; color: #CCC !important; border: 1px solid #333 !important; border-radius: 15px !important; transition: 0.3s !important;}
button.secondary:hover { border-color: #D4AF37 !important; color: #FFF !important; }

/* ABAS SUPERIORES */
.tabs { margin-top: 0 !important; border: none !important; width: 100vw !important;}
.tab-nav { background: #000 !important; border-bottom: 1px solid #111 !important; padding: 10px 0 !important; justify-content: center !important; gap: 10px !important;}
.tab-nav button { background: transparent !important; color: #666 !important; border: none !important; border-radius: 20px !important; padding: 8px 15px !important; font-size: 14px !important;}
.tab-nav button.selected { color: #D4AF37 !important; background: #0A0A0A !important; border: 1px solid #222 !important; font-weight: bold;}

/* CHAT CENTRAL */
.chat-container { background: transparent !important; border: none !important; height: 82vh !important; }
.chatbot { background: transparent !important; border: none !important; max-width: 900px !important; margin: 0 auto !important; width: 100% !important; overflow-x: hidden !important;}

/* BALÕES DE MENSAGEM */
.message-wrap { padding: 20px 0 !important; }
.message { border-radius: 20px !important; padding: 15px 20px !important; font-size: 16px !important; line-height: 1.6; max-width: 85% !important; word-wrap: break-word !important;}
.message.user { background: rgba(212, 175, 55, 0.05) !important; border: 1px solid rgba(212, 175, 55, 0.3) !important; color: #FFFFFF !important; margin-left: auto !important; border-bottom-right-radius: 5px !important; }
.message.bot { background: transparent !important; border: none !important; color: #E0E0E0 !important; margin-right: auto !important; padding-left: 0 !important;}

/* A PÍLULA FLUTUANTE DE DIGITAR MENSAGEM */
.chat-container > div:last-child, .chat-container form { 
    background: #0E0E0E !important; border: 1px solid #333 !important; border-radius: 30px !important; 
    padding: 5px 10px !important; box-shadow: 0 10px 30px rgba(0,0,0,0.8) !important; 
    max-width: 900px !important; margin: 0 auto 15px auto !important; width: 95% !important; box-sizing: border-box !important;
}
.chat-container textarea { background: transparent !important; border: none !important; color: #FFF !important; box-shadow: none !important; }
.chat-container textarea:focus { border: none !important; box-shadow: none !important; }

button.primary { background: linear-gradient(145deg, #D4AF37, #AA7C11) !important; color: #000 !important; border: none !important; border-radius: 30px !important; font-weight: bold !important; transition: 0.3s !important;}
button.primary:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(212, 175, 55, 0.4) !important; }
input { background: #111 !important; border: 1px solid #333 !important; border-radius: 15px !important; color: #FFF !important; max-width: 100% !important;}

::-webkit-scrollbar {width: 6px; height: 6px;}
::-webkit-scrollbar-track {background: transparent;}
::-webkit-scrollbar-thumb {background: #D4AF37; border-radius: 10px;}

/* ================= MEDIA QUERIES MOBILE ================= */
@media screen and (max-width: 768px) {
    .logo-title { font-size: 18px !important; letter-spacing: 1px !important;}
    .sidebar { width: 100vw !important; padding: 10px !important; border-right: none !important; }
    
    .chat-container { height: 78vh !important; }
    .message { max-width: 92% !important; font-size: 15px !important; padding: 12px 18px !important; }
    
    .chat-container > div:last-child, .chat-container form { 
        width: 98% !important; 
        max-width: 98% !important; 
        margin: 0 auto 10px auto !important; 
        border-radius: 25px !important;
    }
}
"""

# ==========================================
# 8. CONSTRUÇÃO DA INTERFACE VISUAL
# ==========================================
with gr.Blocks(title="Código de Ouro", theme=tema_ouro, css=CSS_APP) as interface:
    
    id_sessao_atual = gr.State(f"Chat_{datetime.now().strftime('%d%m_%H%M%S')}")

    with gr.Sidebar(open=True):
        gr.HTML(f"""
        <div class="logo-container">
            {TAG_LOGO}
            <h1 class="logo-title">CÓDIGO DE OURO</h1>
        </div>
        """)
        btn_novo = gr.Button("➕ Novo Chat", variant="primary")
        gr.Markdown("### 📜 Histórico de Conversas")
        lista_chats = gr.Dropdown(choices=listar_sessoes_chat(), label="", interactive=True)
        btn_load = gr.Button("Abrir Conversa", variant="secondary")
        btn_atualizar = gr.Button("🔄 Atualizar Histórico", variant="secondary")
        btn_atualizar.click(lambda: gr.update(choices=listar_sessoes_chat()), None, lista_chats)

    with gr.Tabs():
        
        with gr.TabItem("💬 Chat IA"):
            with gr.Accordion("⚙️ Ajustes", open=False):
                with gr.Row():
                    persona = gr.Dropdown(choices=["Assistente Padrão", "Especialista em Marketing", "Analista de Dados"], value="Assistente Padrão", label="Especialidade", scale=2)
                    net = gr.Checkbox(label="🌐 Web", scale=1)
                    btn_exportar = gr.Button("💾 Exportar (Word)", variant="secondary", scale=1)
            
            chat = gr.ChatInterface(
                fn=responder_chat_central, multimodal=True, additional_inputs=[persona, net, id_sessao_atual],
                chatbot=gr.Chatbot(height=700, show_label=False),
                textbox=gr.MultimodalTextbox(placeholder="Envie uma mensagem, foto ou documento...", container=False)
            )
            arq_exportado = gr.File(label="Arquivo", visible=False)
            btn_exportar.click(exportar_conversa_docx, chat.chatbot, arq_exportado).then(lambda: gr.update(visible=True), None, arq_exportado)
            
            btn_load.click(carregar_sessao_chat, lista_chats, [chat.chatbot, id_sessao_atual])
            btn_novo.click(iniciar_novo_chat, None, [chat.chatbot, id_sessao_atual, lista_chats])

        with gr.TabItem("📑 Documentos"):
            with gr.Row():
                files = gr.File(label="Upload (PDF/Excel)", file_count="multiple")
                inst = gr.Textbox(label="Instrução", placeholder="O que devo analisar?")
            btn_doc = gr.Button("Iniciar Análise Profissional", variant="primary")
            res_doc = gr.Textbox(label="Resultado", lines=10)

        with gr.TabItem("🗂️ Cofre"):
            btn_att_gal = gr.Button("🔄 Sincronizar Galeria Visual", variant="primary")
            gal = gr.Gallery(columns=4, height="auto")
            btn_att_gal.click(atualizar_galeria_imagens, None, gal)

# ==========================================
# 9. LANÇAMENTO 
# ==========================================
usuarios = []
for i in ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9"]:
    u = os.environ.get(f"LOGIN_USUARIO{i}") or os.environ.get(f"USUARIO{i}")
    s = os.environ.get(f"LOGIN_SENHA{i}") or os.environ.get(f"SENHA{i}")
    if u and s:
        usuarios.append((u, s))

interface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 10000)),
    auth=usuarios if usuarios else None,
    auth_message=LOGIN_HACK, 
    js="() => { document.body.classList.add('dark'); }",
    head=PWA_HEAD
)
