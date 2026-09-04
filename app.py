import gradio as gr
import pandas as pd
import os
import docx
import pdfplumber
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

embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=chave_hf, model_name="sentence-transformers/all-MiniLM-L6-v2")

# ==========================================
# 2. DIRETÓRIOS DO SISTEMA
# ==========================================
DIRETORIO = "./AuraFull_Core"
DIR_CHROMA = f"{DIRETORIO}/Banco_Vetorial"
DIR_CASOS = f"{DIRETORIO}/Arquivos_Salvos"
DIR_MIDIA = f"{DIRETORIO}/Midias_Geradas"
DIR_CHATS = f"{DIRETORIO}/Historico"

for d in [DIRETORIO, DIR_CASOS, DIR_MIDIA, DIR_CHATS]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 3. GESTÃO DE SESSÕES
# ==========================================
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
# 4. EXTRAÇÃO E MOTORES MULTIMÍDIA
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

def motor_gerar_imagem(prompt_desc, proporcao="Vertical"):
    try:
        w, h = (1080, 1920) if "vertical" in proporcao.lower() else (1920, 1080) if "horizontal" in proporcao.lower() else (1024, 1024)
        prompt_clean = urllib.parse.quote(prompt_desc.strip())
        url = f"https://image.pollinations.ai/prompt/{prompt_clean}?width={w}&height={h}&nologo=true&seed={int(time.time())}"
        nome_arq = f"{DIR_MIDIA}/Img_{datetime.now().strftime('%H%M%S')}.jpg"
        urllib.request.urlretrieve(url, nome_arq)
        return nome_arq
    except: return None

def motor_editar_imagem(caminho_imagem, prompt_edicao):
    try:
        res = cliente_hf.image_to_image(image=caminho_imagem, prompt=prompt_edicao, model="timbrooks/instruct-pix2pix")
        caminho_saida = f"{DIR_MIDIA}/Edit_{datetime.now().strftime('%H%M%S')}.jpg"
        res.save(caminho_saida)
        return caminho_saida
    except: return None

def motor_gerar_audio(texto):
    try:
        cam_txt = f"{DIR_MIDIA}/temp_{datetime.now().strftime('%H%M%S')}.txt"
        cam_audio = f"{DIR_MIDIA}/Voz_{datetime.now().strftime('%H%M%S')}.mp3"
        with open(cam_txt, "w", encoding="utf-8") as f: f.write(texto[:2500].replace('*', ''))
        os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{cam_txt}" --write-media "{cam_audio}"')
        return cam_audio
    except: return None

def motor_gerar_video(prompt_cena, imagem_base=None):
    try:
        if imagem_base: return Client("multimodalart/stable-video-diffusion", hf_token=chave_hf).predict(imagem_base, api_name="/video")
        else: return Client("multimodalart/zeroscope-v2", hf_token=chave_hf).predict(prompt_cena[:150], api_name="/infer")
    except: return None

# ==========================================
# 5. O CÉREBRO DA AURAFULL
# ==========================================
def responder_chat_central(mensagem, historico, persona, usar_internet, id_sessao):
    texto_usuario = mensagem.get("text", "") if isinstance(mensagem, dict) else str(mensagem)
    arquivos = mensagem.get("files", []) if isinstance(mensagem, dict) else []
    contexto_extra = ""
    imagens_anexadas = []
    
    if arquivos: yield "⏳ Lendo arquivos..."
    
    for arq in arquivos:
        ext = arq.lower()
        if ext.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            imagens_anexadas.append(arq)
        else:
            contexto_extra += f"\n[DOCUMENTO]:\n{extrair_texto(arq)}\n"
            
    if usar_internet and texto_usuario:
        yield "🌐 Pesquisando na web..."
        try:
            resultados = DDGS().text(texto_usuario, max_results=4)
            contexto_extra += "\n\n[DADOS WEB]:\n" + "\n".join([f"{r['title']} - {r['body']}" for r in resultados])
        except: pass

    yield "✨ Processando..."

    sys_prompt = f"""Você é a IA "AuraFull", operando com o perfil {persona}.
Comporte-se como um assistente de altíssimo nível, claro, preciso e direto, similar ao Google Gemini.
Formate suas respostas em Markdown limpo.
PODERES DE GERAÇÃO MULTIMÍDIA (Acione usando os comandos exatos se o usuário pedir):
1. IMAGEM: [AÇÃO_IMAGEM: prompt detalhado em inglês | vertical]
2. EDIÇÃO: [AÇÃO_EDITAR_IMAGEM: comando em inglês]
3. VÍDEO: [AÇÃO_VIDEO: cena em inglês]
4. ÁUDIO: [AÇÃO_AUDIO: texto falado em pt-BR]"""

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
        conteudo_multimodal = [{"type": "text", "text": texto_final if texto_final else "O que há nesta imagem?"}]
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
            yield re.sub(r'\[AÇÃO_\w+:.*?\]', '✨ Criando mídia...', resposta_acumulada)

    anexos_html = ""
    match_img = re.search(r'\[AÇÃO_IMAGEM:\s*(.*?)(?:\|\s*(\w+))?\]', resposta_acumulada)
    if match_img:
        prompt_i = match_img.group(1).strip()
        prop_i = "Vertical" if (match_img.group(2) and "vertical" in match_img.group(2).lower()) else "Quadrado"
        cam_gerada = motor_gerar_imagem(prompt_i, prop_i)
        if cam_gerada:
            b64_img = encode_file_b64(cam_gerada)
            anexos_html += f"\n\n<img src='data:image/jpeg;base64,{b64_img}' style='max-width:100%; border-radius:16px; margin-top:10px;' />\n"

    match_edit = re.search(r'\[AÇÃO_EDITAR_IMAGEM:\s*(.*?)\]', resposta_acumulada)
    if match_edit and imagens_anexadas:
        prompt_e = match_edit.group(1).strip()
        cam_edit = motor_editar_imagem(imagens_anexadas[-1], prompt_e)
        if cam_edit:
            b64_img = encode_file_b64(cam_edit)
            anexos_html += f"\n\n<img src='data:image/jpeg;base64,{b64_img}' style='max-width:100%; border-radius:16px; margin-top:10px;' />\n"

    match_aud = re.search(r'\[AÇÃO_AUDIO:\s*(.*?)\]', resposta_acumulada)
    if match_aud:
        texto_loc = match_aud.group(1).strip()
        cam_aud = motor_gerar_audio(texto_loc)
        if cam_aud:
            b64_aud = encode_file_b64(cam_aud)
            anexos_html += f"\n\n<audio controls src='data:audio/mp3;base64,{b64_aud}' style='width:100%; margin-top:10px;'></audio>\n"

    match_vid = re.search(r'\[AÇÃO_VIDEO:\s*(.*?)\]', resposta_acumulada)
    if match_vid:
        prompt_v = match_vid.group(1).strip()
        img_referencia = imagens_anexadas[-1] if imagens_anexadas else None
        cam_vid = motor_gerar_video(prompt_v, img_referencia)
        if cam_vid:
            b64_vid = encode_file_b64(cam_vid)
            anexos_html += f"\n\n<video controls style='max-width:100%; border-radius:16px; margin-top:10px;' src='data:video/mp4;base64,{b64_vid}'></video>\n"

    resposta_final_limpa = re.sub(r'\[AÇÃO_\w+:.*?\]', '', resposta_acumulada).strip() + anexos_html
    yield resposta_final_limpa

    try:
        sessao_alvo = id_sessao if (id_sessao and id_sessao != "Nenhuma conversa salva") else f"Chat_{datetime.now().strftime('%d%m_%H%M%S')}"
        arq_sessao = f"{DIR_CHATS}/{sessao_alvo}.json"
        historico_atual = []
        if os.path.exists(arq_sessao):
            with open(arq_sessao, "r", encoding="utf-8") as f: historico_atual = json.load(f)
        historico_atual.append({"role": "user", "content": texto_usuario if texto_usuario else "[Arquivo]"})
        historico_atual.append({"role": "assistant", "content": resposta_final_limpa})
        with open(arq_sessao, "w", encoding="utf-8") as f: json.dump(historico_atual, f, ensure_ascii=False, indent=4)
    except: pass

def exportar_conversa_docx(historico):
    if not historico: return None
    pasta = f"{DIR_CASOS}/Exportacoes_{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(pasta, exist_ok=True)
    cam_word = f"{pasta}/Conversa_AuraFull.docx"
    doc = docx.Document()
    doc.add_heading('Registro AuraFull', 0)
    for item in historico:
        if isinstance(item, dict):
            autor = "Você:" if item.get("role") == "user" else "AuraFull:"
            doc.add_heading(autor, level=2)
            doc.add_paragraph(re.sub(r'<.*?>', '', item.get("content", "")))
    doc.save(cam_word)
    return cam_word

def gerar_dossie_lote(arquivos, instrucao, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Instrução necessária.", None
    try:
        progresso(0.1, desc="Lendo arquivos...")
        pasta = f"{DIR_CASOS}/Analise_{datetime.now().strftime('%H%M')}"
        os.makedirs(pasta, exist_ok=True)
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for arq in arquivos:
                txt = extrair_texto(arq)
                banco.add_texts([f"FONTE:\n{c}" for c in fatiador.split_text(txt)])
            
        progresso(0.5, desc="AuraFull processando...")
        contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        prompt = f"DADOS:\n{contexto}\n\nINSTRUÇÃO: {instrucao}"
        resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ, max_tokens=4000).choices[0].message.content
        
        cam_word = f"{pasta}/Relatorio.docx"
        doc = docx.Document()
        doc.add_paragraph(resposta)
        doc.save(cam_word)
        return "✅ Análise Concluída!", cam_word
    except Exception as e: return f"Erro: {e}", None

# ==========================================
# 6. ARQUITETURA DE DESIGN GEMINI CLONE (AURAFULL)
# ==========================================
# Cores rigorosamente exatas ao Dark Mode do Google Gemini
tema_aura = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
).set(
    body_background_fill="#131314",
    body_background_fill_dark="#131314",
    background_fill_primary="#131314",
    background_fill_primary_dark="#131314",
    background_fill_secondary="#1e1f20",
    background_fill_secondary_dark="#1e1f20",
    block_background_fill="#131314",
    block_background_fill_dark="#131314",
    border_color_primary="transparent",
    border_color_primary_dark="transparent",
    body_text_color="#e3e3e3",
    body_text_color_dark="#e3e3e3",
    body_text_color_subdued="#c4c7c5",
    body_text_color_subdued_dark="#c4c7c5",
    button_primary_background_fill="#1e1f20",
    button_primary_background_fill_dark="#1e1f20",
    button_primary_text_color="#e3e3e3",
    button_primary_text_color_dark="#e3e3e3",
    button_secondary_background_fill="#131314",
    button_secondary_background_fill_dark="#131314",
    button_secondary_text_color="#c4c7c5",
    button_secondary_text_color_dark="#c4c7c5",
    block_radius="24px"
)

PWA_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="theme-color" content="#131314">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="AuraFull">
"""

LOGIN_HACK = """
<style>
    body, main { background-color: #131314 !important; color: #e3e3e3 !important; font-family: 'Inter', sans-serif;}
    form { background: #1e1f20 !important; border-radius: 24px !important; padding: 40px !important; max-width: 90% !important; margin: auto !important; width: 400px;}
    button.primary { background: #a8c7fa !important; color: #131314 !important; font-weight: 500 !important; border-radius: 24px !important; border: none !important; font-size: 15px !important; margin-top: 15px !important; padding: 10px !important;}
    input { background-color: #131314 !important; border: none !important; border-radius: 12px !important; color: #e3e3e3 !important; padding: 12px !important; width: 100%;}
    form h2 { display: none !important; }
</style>
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color: #e3e3e3; font-size: 28px; font-weight: 500; margin: 0;">AuraFull</h1>
    <p style="color: #c4c7c5; font-size: 14px; margin-top: 5px;">Faça login para continuar</p>
</div>
"""

CSS_AURA = """
/* Reset total para remover o Gradio */
footer { display: none !important; }
.gradio-container { background-color: #131314 !important; border: none !important; }

/* Sidebar Esquerda (Menus) */
.sidebar { background-color: #131314 !important; border-right: none !important; padding: 15px !important; }

/* Botões da Sidebar estilo Gemini */
button.primary, button.secondary { border: none !important; border-radius: 50px !important; padding: 10px 18px !important; text-align: left !important; justify-content: flex-start !important; font-size: 14px !important; transition: 0.2s !important; box-shadow: none !important;}
button.primary:hover, button.secondary:hover { background-color: #1e1f20 !important; color: #e3e3e3 !important; }

/* Abas (Tabs) Invisíveis para simular navegação fluída */
.tabs { border: none !important; background: transparent !important; }
.tab-nav { background: transparent !important; border: none !important; margin-bottom: 10px !important; gap: 15px !important; justify-content: center !important;}
.tab-nav button { background: transparent !important; border: none !important; color: #c4c7c5 !important; font-size: 14px !important; font-weight: 500 !important; padding: 8px 16px !important; border-radius: 8px !important;}
.tab-nav button.selected { color: #a8c7fa !important; background-color: rgba(168, 199, 250, 0.08) !important; }
.tab-nav button:hover { background-color: rgba(255, 255, 255, 0.04) !important; }

/* Balões do Chat (Idêntico ao Gemini) */
.chatbot { background: transparent !important; border: none !important; }
.message-wrap { padding: 5px 0 !important; }
.message { font-size: 15px !important; padding: 12px 18px !important; line-height: 1.5 !important; box-shadow: none !important; border: none !important; }
.message.user { background-color: #1e1f20 !important; color: #e3e3e3 !important; border-radius: 24px !important; margin-left: auto !important; max-width: 75% !important; border-bottom-right-radius: 4px !important; }
.message.bot { background-color: transparent !important; color: #e3e3e3 !important; margin-right: auto !important; padding-left: 0 !important; max-width: 90% !important; }

/* Input Pílula do Gemini */
.chat-container > div:last-child, .chat-container form { background-color: #1e1f20 !important; border: none !important; border-radius: 32px !important; padding: 4px 16px !important; margin: 0 auto 10px auto !important; max-width: 820px !important; box-shadow: none !important; }
.chat-container textarea { background: transparent !important; border: none !important; color: #e3e3e3 !important; font-size: 15px !important; resize: none !important; padding: 14px 50px 14px 5px !important; box-shadow: none !important;}
.chat-container textarea:focus { outline: none !important; box-shadow: none !important; border: none !important; }
.chat-container textarea::placeholder { color: #c4c7c5 !important; font-weight: 400 !important; }

/* Oculta Labels padrão do Gradio */
.gr-box > label { display: none !important; }

/* Dropdowns e Checkboxes */
.gradio-dropdown { background: transparent !important; border: none !important; }
.gradio-dropdown input { background-color: #1e1f20 !important; color: #e3e3e3 !important; border: none !important; border-radius: 12px !important; }
.dropdown-menu, .options { background-color: #1e1f20 !important; border: 1px solid #333 !important; border-radius: 12px !important; }
input[type="checkbox"] { accent-color: #a8c7fa !important; }

/* Microfone Injetado no Input */
#mic-aura-btn { position: absolute; right: 55px; top: 50%; transform: translateY(-50%); background: transparent; border: none; color: #c4c7c5; cursor: pointer; padding: 8px; border-radius: 50%; transition: 0.2s; z-index: 999; display: flex; align-items: center; justify-content: center; }
#mic-aura-btn:hover { background: rgba(255, 255, 255, 0.08); color: #e3e3e3; }
.mic-aura-active { color: #ff5555 !important; animation: pulse-aura 1.5s infinite; }
@keyframes pulse-aura { 0% { transform: translateY(-50%) scale(1); opacity: 1; } 50% { transform: translateY(-50%) scale(1.1); opacity: 0.7; } 100% { transform: translateY(-50%) scale(1); opacity: 1; } }

/* Texto Aviso (Disclaimer) */
.disclaimer { text-align: center; color: #c4c7c5; font-size: 11px; margin-top: 5px; margin-bottom: 10px; }

@media screen and (max-width: 768px) {
    .chat-container > div:last-child, .chat-container form { width: 95% !important; max-width: 95% !important; border-radius: 28px !important; }
    .message.user { max-width: 85% !important; }
}
"""

JS_AURA = """
function() {
    document.body.classList.add('dark');
    
    function injectAuraMic() {
        const textareas = document.querySelectorAll('.chat-container textarea');
        if (textareas.length === 0) return;
        
        const textarea = textareas[textareas.length - 1];
        const parent = textarea.parentElement;
        
        if (parent && !parent.querySelector('#mic-aura-btn')) {
            parent.style.position = 'relative';
            
            let btn = document.createElement('button');
            btn.id = 'mic-aura-btn';
            btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>';
            parent.appendChild(btn);

            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechRecognition) {
                const recognition = new SpeechRecognition();
                recognition.lang = 'pt-BR';
                recognition.continuous = false;
                let isRecording = false;

                btn.onclick = (e) => {
                    e.preventDefault();
                    if (isRecording) recognition.stop();
                    else recognition.start();
                };

                recognition.onstart = () => {
                    isRecording = true;
                    btn.classList.add('mic-aura-active');
                    textarea.placeholder = "Ouvindo...";
                };

                recognition.onresult = (event) => {
                    let text = event.results[0][0].transcript;
                    textarea.value = textarea.value ? textarea.value + ' ' + text : text;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                };

                recognition.onend = () => { 
                    isRecording = false; 
                    btn.classList.remove('mic-aura-active'); 
                    textarea.placeholder = "Pergunte à AuraFull";
                };
                recognition.onerror = () => { 
                    isRecording = false; 
                    btn.classList.remove('mic-aura-active'); 
                    textarea.placeholder = "Pergunte à AuraFull";
                };
            } else {
                btn.style.display = 'none';
            }
        }
    }
    setTimeout(injectAuraMic, 1000);
    setInterval(injectAuraMic, 2000);
}
"""

# ==========================================
# 7. INTERFACE AURAFULL (Clone Nível OpenAI/Google)
# ==========================================
with gr.Blocks(title="AuraFull", theme=tema_aura, css=CSS_AURA, head=PWA_HEAD, js=JS_AURA, fill_height=True) as interface:
    id_sessao_atual = gr.State(f"Chat_{datetime.now().strftime('%d%m_%H%M%S')}")

    with gr.Row():
        # A BARRA LATERAL (IDÊNTICA AO GEMINI)
        with gr.Column(scale=2, min_width=260, elem_classes="sidebar"):
            gr.HTML("""
            <div style="margin-bottom: 25px; padding-left: 5px;">
                <h1 style="color: #e3e3e3; font-size: 20px; font-weight: 500; letter-spacing: -0.5px; margin: 0;">✨ AuraFull</h1>
            </div>
            """)
            
            btn_novo = gr.Button("➕ Novo chat", variant="secondary")
            
            gr.HTML("<br><div style='color: #c4c7c5; font-size: 12px; margin-left: 10px; margin-bottom: 8px;'>Recentes</div>")
            lista_chats = gr.Dropdown(choices=listar_sessoes_chat(), interactive=True)
            with gr.Row():
                btn_load = gr.Button("Abrir", variant="secondary")
                btn_atualizar = gr.Button("Atualizar", variant="secondary")
            btn_atualizar.click(lambda: gr.update(choices=listar_sessoes_chat()), None, lista_chats)

            gr.HTML("<br><div style='color: #c4c7c5; font-size: 12px; margin-left: 10px; margin-bottom: 8px;'>Ferramentas</div>")
            persona = gr.Dropdown(choices=["Assistente Padrão", "Criador de Conteúdo", "Analista de Dados"], value="Assistente Padrão")
            net = gr.Checkbox(label="Pesquisa na Web Ativa", value=False)
            btn_exportar = gr.Button("Baixar PDF/Docs", variant="secondary")

        # A ÁREA DE CHAT PRINCIPAL
        with gr.Column(scale=8):
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    chat = gr.ChatInterface(
                        fn=responder_chat_central, multimodal=True, additional_inputs=[persona, net, id_sessao_atual],
                        chatbot=gr.Chatbot(show_label=False), 
                        textbox=gr.MultimodalTextbox(placeholder="Pergunte à AuraFull", container=False)
                    )
                    gr.HTML("<div class='disclaimer'>AuraFull pode cometer erros. Considere verificar as informações importantes.</div>")
                    
                    arq_exportado = gr.File(visible=False)
                    btn_exportar.click(exportar_conversa_docx, chat.chatbot, arq_exportado).then(lambda: gr.update(visible=True), None, arq_exportado)
                    btn_load.click(carregar_sessao_chat, lista_chats, [chat.chatbot, id_sessao_atual])
                    btn_novo.click(iniciar_novo_chat, None, [chat.chatbot, id_sessao_atual, lista_chats])

                with gr.TabItem("Análise de Documentos"):
                    gr.HTML("<br><h2 style='text-align: center; color: #e3e3e3; font-weight: 400;'>Análise de Dados e Documentos</h2><br>")
                    with gr.Row():
                        files = gr.File(label="Upload (PDF, Word, Excel)", file_count="multiple")
                        with gr.Column():
                            inst = gr.Textbox(placeholder="O que você precisa descobrir nestes documentos?", lines=4)
                            btn_doc = gr.Button("Analisar", variant="secondary")
                    res_doc = gr.Textbox(placeholder="O relatório gerado aparecerá aqui...", lines=12)
                    btn_doc.click(gerar_dossie_lote, [files, inst], [res_doc])

                with gr.TabItem("Galeria"):
                    gr.HTML("<br><h2 style='text-align: center; color: #e3e3e3; font-weight: 400;'>Biblioteca de Mídias Geradas</h2><br>")
                    btn_att_gal = gr.Button("Atualizar Biblioteca", variant="secondary")
                    gal = gr.Gallery(columns=4, height="auto")
                    btn_att_gal.click(atualizar_galeria_imagens, None, gal)

# ==========================================
# 8. LANÇAMENTO 
# ==========================================
usuarios = []
for i in ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9"]:
    u = os.environ.get(f"LOGIN_USUARIO{i}") or os.environ.get(f"USUARIO{i}")
    s = os.environ.get(f"LOGIN_SENHA{i}") or os.environ.get(f"SENHA{i}")
    if u and s: usuarios.append((u, s))

interface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 10000)),
    auth=usuarios if usuarios else None, 
    auth_message=LOGIN_HACK
)
