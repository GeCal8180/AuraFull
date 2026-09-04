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
# 2.5. RADAR DA LOGOMARCA OFICIAL
# ==========================================
TAG_LOGO = ""
FAVICON_TAGS = ""
caminho_logo = None

for root_dir, _, files in os.walk("."):
    for f in files:
        if "chamariz" in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            caminho_logo = os.path.join(root_dir, f)
            break
    if caminho_logo: break

if caminho_logo:
    with open(caminho_logo, "rb") as f:
        b64_logo = base64.b64encode(f.read()).decode('utf-8')
        TAG_LOGO = f'<img src="data:image/png;base64,{b64_logo}" style="max-height: 70px; margin: 0 auto; display: block;" alt="Código de Ouro" />'
        FAVICON_TAGS = f'<link rel="icon" type="image/png" href="data:image/png;base64,{b64_logo}">'
else:
    TAG_LOGO = '<div style="color:#D4AF37; text-align:center; font-weight:bold; font-size: 20px;">CÓDIGO DE OURO</div>'

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
# 5. O CHAT AGÊNTICO
# ==========================================
def responder_chat_central(mensagem, historico, persona, usar_internet, id_sessao):
    texto_usuario = mensagem.get("text", "") if isinstance(mensagem, dict) else str(mensagem)
    arquivos = mensagem.get("files", []) if isinstance(mensagem, dict) else []
    contexto_extra = ""
    imagens_anexadas = []
    
    yield "⏳ *Sincronizando ambiente...*"
    
    for arq in arquivos:
        ext = arq.lower()
        if ext.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            imagens_anexadas.append(arq)
            yield f"👁️ *Analisando imagem...*"
        else:
            yield f"📄 *Lendo documento...*"
            contexto_extra += f"\n[DOCUMENTO]:\n{extrair_texto(arq)}\n"
            
    if usar_internet and texto_usuario:
        yield "🌐 *Buscando na Web em tempo real...*"
        try:
            resultados = DDGS().text(texto_usuario, max_results=4)
            contexto_extra += "\n\n[DADOS DA WEB]:\n" + "\n".join([f"{r['title']} - {r['body']}" for r in resultados])
        except: pass

    yield "🧠 *Processando raciocínio...*"

    sys_prompt = f"""Você é a IA "Código de Ouro" operando no perfil {persona}.
Aja de forma inteligente e direta.
PODERES DE MÍDIA:
1. GERAR IMAGEM: [AÇÃO_IMAGEM: prompt detalhado em inglês | vertical]
2. EDITAR IMAGEM: [AÇÃO_EDITAR_IMAGEM: instrução em inglês]
3. GERAR VÍDEO: [AÇÃO_VIDEO: cena em inglês]
4. GERAR ÁUDIO: [AÇÃO_AUDIO: texto falado em pt-BR]"""

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
        conteudo_multimodal = [{"type": "text", "text": texto_final if texto_final else "Descreva a imagem."}]
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
            yield re.sub(r'\[AÇÃO_\w+:.*?\]', '⚙️ *(Gerando ativo multimídia...)*', resposta_acumulada)

    anexos_html = ""
    match_img = re.search(r'\[AÇÃO_IMAGEM:\s*(.*?)(?:\|\s*(\w+))?\]', resposta_acumulada)
    if match_img:
        prompt_i = match_img.group(1).strip()
        prop_i = "Vertical" if (match_img.group(2) and "vertical" in match_img.group(2).lower()) else "Quadrado"
        cam_gerada = motor_gerar_imagem(prompt_i, prop_i)
        if cam_gerada:
            b64_img = encode_file_b64(cam_gerada)
            anexos_html += f"\n\n**🖼️ Imagem:**\n<img src='data:image/jpeg;base64,{b64_img}' style='max-width:100%; border-radius:12px; margin-top:10px;' />\n"

    match_edit = re.search(r'\[AÇÃO_EDITAR_IMAGEM:\s*(.*?)\]', resposta_acumulada)
    if match_edit and imagens_anexadas:
        prompt_e = match_edit.group(1).strip()
        cam_edit = motor_editar_imagem(imagens_anexadas[-1], prompt_e)
        if cam_edit:
            b64_img = encode_file_b64(cam_edit)
            anexos_html += f"\n\n**✨ Edição:**\n<img src='data:image/jpeg;base64,{b64_img}' style='max-width:100%; border-radius:12px; margin-top:10px;' />\n"

    match_aud = re.search(r'\[AÇÃO_AUDIO:\s*(.*?)\]', resposta_acumulada)
    if match_aud:
        texto_loc = match_aud.group(1).strip()
        cam_aud = motor_gerar_audio(texto_loc)
        if cam_aud:
            b64_aud = encode_file_b64(cam_aud)
            anexos_html += f"\n\n**🔊 Áudio:**\n<audio controls src='data:audio/mp3;base64,{b64_aud}' style='width:100%; margin-top:10px;'></audio>\n"

    match_vid = re.search(r'\[AÇÃO_VIDEO:\s*(.*?)\]', resposta_acumulada)
    if match_vid:
        prompt_v = match_vid.group(1).strip()
        img_referencia = imagens_anexadas[-1] if imagens_anexadas else None
        cam_vid = motor_gerar_video(prompt_v, img_referencia)
        if cam_vid:
            b64_vid = encode_file_b64(cam_vid)
            anexos_html += f"\n\n**🎥 Vídeo:**\n<video controls style='max-width:100%; border-radius:12px; margin-top:10px;' src='data:video/mp4;base64,{b64_vid}'></video>\n"

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
    cam_word = f"{pasta}/Conversa.docx"
    doc = docx.Document()
    doc.add_heading('Registro de Conversa', 0)
    for item in historico:
        if isinstance(item, dict):
            autor = "Você:" if item.get("role") == "user" else "IA:"
            doc.add_heading(autor, level=2)
            doc.add_paragraph(re.sub(r'<.*?>', '', item.get("content", "")))
    doc.save(cam_word)
    return cam_word

def gerar_dossie_lote(arquivos, instrucao, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Instrução necessária.", None
    try:
        progresso(0.1, desc="Processando...")
        pasta = f"{DIR_CASOS}/Analise_{datetime.now().strftime('%H%M')}"
        os.makedirs(pasta, exist_ok=True)
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for arq in arquivos:
                txt = extrair_texto(arq)
                banco.add_texts([f"FONTE:\n{c}" for c in fatiador.split_text(txt)])
            
        progresso(0.5, desc="Analisando...")
        contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        prompt = f"DADOS:\n{contexto}\n\nINSTRUÇÃO: {instrucao}"
        resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ, max_tokens=4000).choices[0].message.content
        
        cam_word = f"{pasta}/Relatorio.docx"
        doc = docx.Document()
        doc.add_paragraph(resposta)
        doc.save(cam_word)
        return "✅ Sucesso!", cam_word
    except Exception as e: return f"Erro: {e}", None

# ==========================================
# 6. TEMA "GEMINI CLONE" (Zero Conflitos de CSS)
# ==========================================
# Toda a configuração de cores agora acontece de forma NATIVA pelo Gradio.
# Isso impede o "Erro" da tela e garante que dropdowns nunca fiquem brancos.
tema_gemini = gr.themes.Default(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
).set(
    body_background_fill="#131314",
    body_background_fill_dark="#131314",
    block_background_fill="#1E1F22",
    block_background_fill_dark="#1E1F22",
    border_color_primary="#333538",
    border_color_primary_dark="#333538",
    body_text_color="#E3E3E3",
    body_text_color_dark="#E3E3E3",
    block_title_text_color="#D4AF37",
    block_title_text_color_dark="#D4AF37",
    block_label_text_color="#D4AF37",
    button_primary_background_fill="#D4AF37",
    button_primary_background_fill_dark="#D4AF37",
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#000000",
    checkbox_background_color_selected="#D4AF37",
    checkbox_background_color_selected_dark="#D4AF37",
    block_radius="16px"
)

PWA_HEAD = f"""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="theme-color" content="#131314">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Código Ouro">
{FAVICON_TAGS}
"""

LOGIN_HACK = """
<style>
    body, main { background-color: #131314 !important; color: #fff !important; font-family: 'Inter', sans-serif;}
    form { background: #1E1F22 !important; border: 1px solid #333538 !important; border-radius: 24px !important; padding: 40px !important; max-width: 90% !important; margin: auto !important; width: 400px;}
    button.primary { background: #D4AF37 !important; color: #000 !important; font-weight: bold !important; border-radius: 12px !important; border: none !important; font-size: 16px !important; margin-top: 15px !important; padding: 12px !important;}
    input { background-color: #131314 !important; border: 1px solid #333538 !important; border-radius: 10px !important; color: #D4AF37 !important; padding: 12px !important; width: 100%;}
    form h2 { display: none !important; }
</style>
<div style="text-align: center; margin-bottom: 30px;">
    [LOGO_PLACEHOLDER]
    <h1 style="color: #D4AF37; font-size: 24px; font-weight: 900; margin: 10px 0 0 0;">CÓDIGO DE OURO</h1>
</div>
""".replace("[LOGO_PLACEHOLDER]", TAG_LOGO)

# Apenas estética visual. SEM regras de layout (width, height, flex, box-sizing) que quebravam o Gradio.
CSS_GEMINI = """
footer { display: none !important; }

/* Balões de Chat e Input do Gemini */
.message { border-radius: 18px !important; font-size: 15px !important; padding: 12px 18px !important; }
.message.user { background-color: #1E1F22 !important; border: 1px solid #333538 !important; }
.message.bot { background-color: transparent !important; border: none !important; }
.chat-container form { background-color: #1E1F22 !important; border: 1px solid #333538 !important; border-radius: 24px !important; box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;}

/* Esconde labels desncessárias */
.gr-box > label { display: none !important; }

/* Estilo do Botão Microfone Isolado */
#btn-mic-gemini {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background-color: #1E1F22;
    border: 1px solid #333538;
    color: #D4AF37;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
    z-index: 9999;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: 0.3s;
}
#btn-mic-gemini:hover { background-color: #333538; }
.mic-active { background-color: #ff4444 !important; border-color: #ff4444 !important; color: #fff !important; animation: mic-pulse 1.5s infinite; }
@keyframes mic-pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.6); } 70% { box-shadow: 0 0 0 15px rgba(255, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); } }

@media screen and (max-width: 768px) {
    #btn-mic-gemini { bottom: 20px; right: 20px; width: 50px; height: 50px; }
}
"""

JS_GEMINI = """
function() {
    function injectMic() {
        if (document.getElementById('btn-mic-gemini')) return;
        
        let btn = document.createElement('button');
        btn.id = 'btn-mic-gemini';
        btn.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>';
        document.body.appendChild(btn);

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
                btn.classList.add('mic-active');
            };

            recognition.onresult = (event) => {
                let text = event.results[0][0].transcript;
                let activeInput = null;
                const textareas = document.querySelectorAll('textarea');
                textareas.forEach(ta => { if (ta.offsetParent !== null) activeInput = ta; });
                
                if (activeInput) {
                    activeInput.value = activeInput.value ? activeInput.value + ' ' + text : text;
                    activeInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };

            recognition.onend = () => { isRecording = false; btn.classList.remove('mic-active'); };
            recognition.onerror = () => { isRecording = false; btn.classList.remove('mic-active'); };
        } else {
            btn.style.display = 'none';
        }
    }
    setTimeout(injectMic, 1000);
    setInterval(injectMic, 2000);
}
"""

# ==========================================
# 7. ESTRUTURA VISUAL (SIMPLES, ESTÁVEL, NATIVA)
# ==========================================
with gr.Blocks(title="Código de Ouro", theme=tema_gemini, css=CSS_GEMINI, head=PWA_HEAD, js=JS_GEMINI, fill_height=True) as interface:
    id_sessao_atual = gr.State(f"Chat_{datetime.now().strftime('%d%m_%H%M%S')}")

    # A Sidebar Oficial do Gradio. Ela recolhe e cria o menu de celular automaticamente!
    with gr.Sidebar():
        gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            {TAG_LOGO}
        </div>
        """)
        btn_novo = gr.Button("➕ Novo Atendimento", variant="primary")
        
        gr.HTML("<h3 style='color:#D4AF37; margin-top:20px; font-size:14px;'>⚙️ CONFIGURAÇÕES</h3>")
        persona = gr.Dropdown(choices=["Assistente Padrão", "Gênio do Marketing", "Analista de Dados"], value="Assistente Padrão", label="Especialidade", interactive=True)
        net = gr.Checkbox(label="Pesquisa Web Integrada", value=False)
        btn_exportar = gr.Button("💾 Exportar Documento")
        
        gr.HTML("<h3 style='color:#D4AF37; margin-top:20px; font-size:14px;'>📜 HISTÓRICO</h3>")
        lista_chats = gr.Dropdown(choices=listar_sessoes_chat(), label="Sessões", interactive=True)
        with gr.Row():
            btn_load = gr.Button("Abrir")
            btn_atualizar = gr.Button("Atualizar")
        btn_atualizar.click(lambda: gr.update(choices=listar_sessoes_chat()), None, lista_chats)

    # Abas limpas sem forçar colunas
    with gr.Tabs():
        with gr.TabItem("💬 Console de IA"):
            chat = gr.ChatInterface(
                fn=responder_chat_central, multimodal=True, additional_inputs=[persona, net, id_sessao_atual],
                chatbot=gr.Chatbot(show_label=False), textbox=gr.MultimodalTextbox(placeholder="Pergunte ao Código de Ouro...", container=False)
            )
            arq_exportado = gr.File(label="Documento Gerado", visible=False)
            btn_exportar.click(exportar_conversa_docx, chat.chatbot, arq_exportado).then(lambda: gr.update(visible=True), None, arq_exportado)
            btn_load.click(carregar_sessao_chat, lista_chats, [chat.chatbot, id_sessao_atual])
            btn_novo.click(iniciar_novo_chat, None, [chat.chatbot, id_sessao_atual, lista_chats])

        with gr.TabItem("📑 Analisador de Documentos"):
            with gr.Row():
                files = gr.File(label="Arquivos (PDF/Excel)", file_count="multiple")
                with gr.Column():
                    inst = gr.Textbox(label="Diretrizes", placeholder="O que eu devo analisar?", lines=3)
                    btn_doc = gr.Button("Processar Dados", variant="primary")
            res_doc = gr.Textbox(label="Relatório", lines=10)
            btn_doc.click(gerar_dossie_lote, [files, inst], [res_doc])

        with gr.TabItem("🗂️ Galeria"):
            btn_att_gal = gr.Button("🔄 Atualizar Mídias", variant="primary")
            gal = gr.Gallery(columns=4, height="auto")
            btn_att_gal.click(atualizar_galeria_imagens, None, gal)

# ==========================================
# 8. LANÇAMENTO SEGURO
# ==========================================
usuarios = []
for i in ["", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9"]:
    u = os.environ.get(f"LOGIN_USUARIO{i}") or os.environ.get(f"USUARIO{i}")
    s = os.environ.get(f"LOGIN_SENHA{i}") or os.environ.get(f"SENHA{i}")
    if u and s: usuarios.append((u, s))

launch_args = {
    "server_name": "0.0.0.0", 
    "server_port": int(os.environ.get("PORT", 10000)),
    "auth": usuarios if usuarios else None, 
    "auth_message": LOGIN_HACK
}

interface.launch(**launch_args)
