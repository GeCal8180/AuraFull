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
from datetime import datetime
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
from gradio_client import Client

# ==========================================
# 1. CHAVES MESTRES E CONEXÕES
# ==========================================
chave_groq = os.environ.get("GROQ_API_KEY")
chave_hf = os.environ.get("HF_TOKEN")

cliente_groq = Groq(api_key=chave_groq)
cliente_hf = InferenceClient(token=chave_hf)

# MOTOR INSTANTÂNEO (Liberado, Imortal e sem Erro 400/404)
MODELO_GROQ = "llama-3.1-8b-instant"
MODELO_VISAO = "llama-3.2-90b-vision-preview"

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=chave_hf, 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# 2. DIRETÓRIOS E LOGS
# ==========================================
DIRETORIO = "./Central_IA_Master"
DIR_CHROMA = f"{DIRETORIO}/Banco_de_Dados_Vetorial"
DIR_CASOS = f"{DIRETORIO}/Projetos_Salvos"
DIR_MIDIA = f"{DIRETORIO}/Midia_Criada"

for d in [DIRETORIO, DIR_CASOS, DIR_MIDIA]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 3. RENDERIZAÇÃO DA LOGO (RADAR ABSOLUTO)
# ==========================================
def renderizar_logo():
    # O Radar: Vasculha todas as pastas ignorando maiúsculas e minúsculas
    caminho_real = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if "chamariz" in f.lower() and "fundo" in f.lower():
                caminho_real = os.path.join(root, f)
                break
        if caminho_real:
            break

    if caminho_real:
        with open(caminho_real, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f'''
        <div style="display: flex; justify-content: center; align-items: center; padding: 10px 0 25px 0;">
            <img src="data:image/jpeg;base64,{encoded}" style="max-width: 80%; filter: drop-shadow(0px 8px 15px rgba(212, 175, 55, 0.5));">
        </div>
        '''
    else:
        return '''
        <div style="text-align: center; margin-bottom: 25px; padding: 20px 10px; border-radius: 15px; background: linear-gradient(145deg, #BF953F, #B38728); box-shadow: 0 10px 25px rgba(212,175,55,0.3);">
            <h1 style="color: #000; font-family: 'Montserrat', sans-serif; font-weight: 900; letter-spacing: 2px; margin: 0; font-size: 18px;">O CÓDIGO</h1>
            <h1 style="color: #000; font-family: 'Montserrat', sans-serif; font-weight: 900; letter-spacing: 2px; margin: 0; font-size: 18px;">DE OURO</h1>
        </div>
        '''

def encode_image(image_path):
    with open(image_path, "rb") as image_file: return base64.b64encode(image_file.read()).decode('utf-8')

def extrair_texto(arquivo):
    caminho = arquivo.name if hasattr(arquivo, 'name') else arquivo
    nome = caminho.lower()
    texto = ""
    try:
        if nome.endswith('.pdf'):
            with pdfplumber.open(caminho) as pdf:
                for p in pdf.pages:
                    txt = p.extract_text()
                    if txt: texto += txt + "\n"
        elif nome.endswith(('.xlsx', '.csv')): texto = (pd.read_excel(caminho) if nome.endswith('.xlsx') else pd.read_csv(caminho)).to_string() 
        elif nome.endswith('.docx'):
            for p in docx.Document(caminho).paragraphs: texto += p.text + "\n"
    except: pass
    return texto

def limpar_banco_de_dados():
    try:
        if os.path.exists(DIR_CHROMA): shutil.rmtree(DIR_CHROMA)
        return "🧹 Memória Neural Reiniciada."
    except Exception as e: return f"Erro: {e}"

# ==========================================
# 4. CHAT ULTRAMODERNO (CORE BLINDADO)
# ==========================================
def responder_chat_multimodal(mensagem, historico, persona, usar_internet):
    try:
        texto_usuario = mensagem.get("text", "") if isinstance(mensagem, dict) else str(mensagem)
        arquivos = mensagem.get("files", []) if isinstance(mensagem, dict) else []
        contexto_extra, imagens = "", []
        
        for arq in arquivos:
            cam = arq["path"] if isinstance(arq, dict) and "path" in arq else (arq if isinstance(arq, str) else getattr(arq, 'name', ''))
            if not cam: continue
            if cam.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): imagens.append(cam)
            else: contexto_extra += f"\n[DOC ANEXADO]:\n{extrair_texto(cam)}\n"
                
        if usar_internet and texto_usuario:
            try:
                resultados = DDGS().text(texto_usuario, max_results=3)
                contexto_extra += "\n\n[DADOS WEB]:\n" + "\n".join([f"Fonte: {r['title']} - Resumo: {r['body']}" for r in resultados])
            except: pass

        sys_prompt = f"Você atua no sistema de elite 'O Código de Ouro' e é um {persona}. Responda com excelência absoluta, foco comercial e tom majestoso."
        mensagens = [{"role": "system", "content": sys_prompt}]
        
        for item in historico:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "")
                if isinstance(content, (tuple, list, dict)): content = "[Mídia Anexada Anteriormente]"
                if content and str(content).strip(): mensagens.append({"role": role, "content": str(content)})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                h_u, h_a = item
                if h_u: mensagens.append({"role": "user", "content": "[Mídia Anexada]" if isinstance(h_u, (tuple, list, dict)) else str(h_u)})
                if h_a: mensagens.append({"role": "assistant", "content": str(h_a)})
        
        texto_final = texto_usuario + contexto_extra
        if imagens and not texto_final.strip(): texto_final = "Analise esta imagem em detalhes."
        elif not imagens and not texto_final.strip(): return "⚠️ Operação cancelada. Insira um comando."

        if imagens:
            conteudo_msg = [{"type": "text", "text": texto_final}]
            for img in imagens: conteudo_msg.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"}})
            mensagens.append({"role": "user", "content": conteudo_msg})
            modelo = MODELO_VISAO
        else:
            mensagens.append({"role": "user", "content": texto_final})
            modelo = MODELO_GROQ

        resposta = cliente_groq.chat.completions.create(messages=mensagens, model=modelo, max_tokens=3000)
        return resposta.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ **Falha de Conexão.** Detalhe técnico: `{str(e)}`"

def exportar_conversa(historico):
    if not historico: return None
    pasta = f"{DIR_CASOS}/Chat_{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(pasta, exist_ok=True)
    cam_word = f"{pasta}/Protocolo_O_Codigo_de_Ouro.docx"
    doc = docx.Document()
    doc.add_heading('Protocolo - O Código de Ouro', 0)
    for item in historico:
        if isinstance(item, dict):
            role = "Comando:" if item.get("role") == "user" else "O Código de Ouro:"
            content = "[Arquivo]" if isinstance(item.get("content"), (tuple, list, dict)) else str(item.get("content"))
            doc.add_heading(role, level=2)
            doc.add_paragraph(content)
    doc.save(cam_word)
    return cam_word

# ==========================================
# 5. DOSSIÊ E MESA DO DIRETOR
# ==========================================
def gerar_dossie(arquivos, instrucao, usar_img, usar_aud, usar_tribunal, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Faltou instrução", None, None, "", ""
    t_inicio = time.time()
    palavras = 0
    try:
        progresso(0.1, desc="Auditando Cofre...")
        pasta = f"{DIR_CASOS}/Projeto_{datetime.now().strftime('%d_%m_%Y__%Hh%M')}"
        os.makedirs(pasta, exist_ok=True)
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for idx, arq in enumerate(arquivos):
                progresso((0.1 + (0.3 * (idx/len(arquivos)))), desc="Mapeando dados...")
                txt = extrair_texto(arq)
                palavras += len(txt.split())
                banco.add_texts([f"[FONTE: {os.path.basename(arq.name)}]\n{c}" for c in fatiador.split_text(txt)])
            
        progresso(0.5, desc="Sintetizando Ouro...")
        contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        regra_imagem = "\nNo final, escreva: [IMAGEM: descreva em INGLÊS uma cena fotorrealista para este conteúdo]" if usar_img else ""
        prompt = f"Você é o analista do sistema 'O Código de Ouro'. {regra_imagem}\nDADOS: {contexto}\nAÇÃO: {instrucao}"
        resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ, max_tokens=4000).choices[0].message.content
        
        cam_img = None
        if usar_img:
            progresso(0.8, desc="Renderizando Arte...")
            match = re.search(r'\[IMAGEM:\s*(.*?)\]', resposta, re.IGNORECASE)
            resposta_limpa = re.sub(r'\[IMAGEM:\s*(.*?)\]', '', resposta, flags=re.IGNORECASE).strip()
            prompt_img = match.group(1).strip() if match else "Minimalist corporate presentation cover, gold accents"
            cam_img = f"{pasta}/Capa_Projeto.jpg"
            cliente_hf.text_to_image(prompt_img, model="black-forest-labs/FLUX.1-schnell").save(cam_img)
        else: resposta_limpa = resposta

        progresso(0.9, desc="Exportando...")
        cam_word = f"{pasta}/Relatorio_Codigo_de_Ouro.docx"
        doc = docx.Document()
        doc.add_heading('Relatório - O Código de Ouro', 0)
        if cam_img: doc.add_picture(cam_img, width=Inches(6.0))
        doc.add_paragraph(resposta_limpa)
        doc.save(cam_word)
        
        progresso(1.0, desc="Operação Finalizada")
        cam_audio = f"{pasta}/Audio_Codigo_Ouro.mp3"
        if usar_aud:
            with open(f"{pasta}/temp.txt", "w", encoding="utf-8") as f: f.write(resposta_limpa[:3000].replace('*', ''))
            os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{pasta}/temp.txt" --write-media "{cam_audio}"')
        return "✅ Auditoria Concluída", cam_word, cam_audio if usar_aud else None, resposta_limpa, f"📊 STATUS: {palavras} palavras auditadas."
    except Exception as e: return f"Erro crítico: {e}", None, None, "", ""

def aprimorar_prompt(sujeito, fundo, estilo, tipo="imagem"):
    if not sujeito: return None
    instrucao = f"Traduza para INGLÊS. Sujeito: {sujeito} | Fundo: {fundo} | Estilo: {estilo}. Adicione: 8k, highly detailed, photorealistic. Responda APENAS o texto em inglês." if tipo=="imagem" else f"Traduza para INGLÊS (MÁX 40 PALAVRAS). Ação: {sujeito} | Fundo: {fundo} | Movimento: {estilo}. Responda APENAS o texto em inglês."
    try:
        r = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": instrucao}], model=MODELO_GROQ, temperature=0.1).choices[0].message.content.strip()
        if ":" in r and len(r.split(":")[0]) < 20: r = r.split(":", 1)[-1].strip()
        return r.replace('"', '')
    except: return f"{sujeito}, {fundo}, {estilo}"

def gerar_imagem_estudio(sujeito, fundo, estilo):
    if not sujeito: return None
    c = f"{DIR_MIDIA}/Img_{datetime.now().strftime('%H%M%S')}.jpg"
    cliente_hf.text_to_image(aprimorar_prompt(sujeito, fundo, estilo, "imagem"), model="black-forest-labs/FLUX.1-schnell").save(c)
    return c

def gerar_video_ia(imagem_base, sujeito, fundo, movimento):
    if imagem_base:
        try: return Client("multimodalart/stable-video-diffusion").predict(imagem_base, api_name="/video"), "✅ Cena animada!"
        except: return None, "⚠️ Motor gráfico congestionado."
    else:
        if not sujeito: return None, "⚠️ Preencha a Ação."
        try: return Client("multimodalart/zeroscope-v2").predict(aprimorar_prompt(sujeito, fundo, movimento, "vídeo"), api_name="/infer"), "✅ Vídeo renderizado!"
        except: return None, "⚠️ Motor gráfico congestionado."

def falar_laudo_estudio(texto):
    if not texto: return None
    cam_txt, cam_audio = f"{DIR_MIDIA}/temp.txt", f"{DIR_MIDIA}/Voz_{datetime.now().strftime('%H%M%S')}.mp3"
    with open(cam_txt, "w", encoding="utf-8") as f: f.write(texto[:3000].replace('*', ''))
    os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{cam_txt}" --write-media "{cam_audio}"')
    return cam_audio

def gerar_backup():
    cam = "./Cofre_Codigo_De_Ouro.zip"
    with zipfile.ZipFile(cam, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(DIRETORIO):
            if "Banco_de_Dados_Vetorial" not in root:
                for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), DIRETORIO))
    return cam, "📦 Arquivos Extraídos"

# ==========================================
# 7. DESIGN DE ALTO CONTRASTE (O CÓDIGO DE OURO)
# ==========================================
tema_ultra = gr.themes.Base(
    font=[gr.themes.GoogleFont("Montserrat"), "sans-serif"],
).set(
    body_background_fill="#050505",
    body_background_fill_dark="#050505",
    body_text_color="#FFFFFF",
    body_text_color_dark="#FFFFFF",
    background_fill_primary="#0A0A0A",
    background_fill_primary_dark="#0A0A0A",
    background_fill_secondary="#111111",
    background_fill_secondary_dark="#111111",
    border_color_primary="#333333",
    border_color_primary_dark="#333333",
    block_background_fill="#0A0A0A",
    block_background_fill_dark="#0A0A0A",
    block_label_text_color="#D4AF37",
    block_label_text_color_dark="#D4AF37",
    block_title_text_color="#D4AF37",
    block_title_text_color_dark="#D4AF37",
    input_background_fill="#141414",
    input_background_fill_dark="#141414",
    input_border_color="#444444",
    input_border_color_dark="#444444",
    button_primary_background_fill="linear-gradient(145deg, #D4AF37, #AA7C11)",
    button_primary_background_fill_dark="linear-gradient(145deg, #D4AF37, #AA7C11)",
    button_primary_text_color="#000000",
    button_secondary_background_fill="#1A1A1A",
    button_secondary_background_fill_dark="#1A1A1A",
    button_secondary_text_color="#D4AF37"
)

css_ultra = """
body, .gradio-container { background-color: #050505 !important; color: #FFFFFF !important; font-family: 'Montserrat', sans-serif !important; }
footer { display: none !important; }
span, p, label, h1, h2, h3, h4, .markdown-text, .chatbot { color: #F3F4F6 !important; }
h3 { color: #D4AF37 !important; }

/* =======================================================
   CORREÇÃO DO LOGIN: Bloqueio do Autofill do Navegador
========================================================== */
input:-webkit-autofill,
input:-webkit-autofill:hover, 
input:-webkit-autofill:focus, 
input:-webkit-autofill:active {
    -webkit-box-shadow: 0 0 0 30px #111111 inset !important;
    -webkit-text-fill-color: #FFFFFF !important;
    transition: background-color 5000s ease-in-out 0s;
}

/* Legibilidade forçada para inputs e caixas de texto */
textarea, input, select, .wrap-inner, .dropdown-menu, .wrap { 
    background-color: #111111 !important; 
    color: #FFFFFF !important; 
    border: 1px solid #333333 !important; 
    border-radius: 12px !important; 
}
.label-wrap span { color: #CCCCCC !important; }

button { text-transform: uppercase; font-weight: 700 !important; letter-spacing: 1px; transition: 0.3s all ease !important; border-radius: 12px !important; color: #FFFFFF !important; }
button:hover { transform: translateY(-2px); }
button.primary { color: #000000 !important; border: none !important; box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4) !important; text-shadow: none !important; }
button.secondary { border: 1px solid #444 !important; background: #111 !important; color: #D4AF37 !important; }
.tabs { background: transparent !important; border: none !important; }
.tab-nav { background: #0A0A0A !important; border-bottom: 2px solid #222 !important; padding: 10px 20px 0 !important; gap: 10px; }
.tab-nav button { color: #888888 !important; padding: 12px 25px !important; border-radius: 10px 10px 0 0 !important; border: none !important; }
.tab-nav button.selected { color: #D4AF37 !important; border-bottom: 3px solid #D4AF37 !important; background: #111111 !important; }
.box-painel { background: #0A0A0A !important; border-radius: 16px !important; padding: 30px !important; border: 1px solid #222 !important; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
.chat-container { border-radius: 16px !important; background: #0A0A0A !important; border: 1px solid #222 !important; }
.message.bot { background: #111111 !important; border-left: 3px solid #D4AF37 !important; color: #FFFFFF !important; }
.message.user { background: #1A1A1A !important; border: 1px solid #333 !important; color: #D4AF37 !important; }
.sidebar { background: #050505 !important; border-right: 1px solid #222 !important; padding: 25px !important; }
"""

forca_dark_mode = "function() { document.body.classList.add('dark'); }"

with gr.Blocks(title="O Código de Ouro", theme=tema_ultra, css=css_ultra, fill_width=True, js=forca_dark_mode) as interface:
    
    with gr.Sidebar(open=True):
        gr.HTML(renderizar_logo())
        
        gr.HTML("<h3 style='text-align: center; color: #D4AF37; text-transform: uppercase; font-size: 10px; letter-spacing: 3px; margin-bottom: 30px;'>Inteligência Suprema</h3>")
        
        gr.Markdown("### ⚙️ Engine Operacional")
        persona_box = gr.Dropdown(choices=["Assessor Gold (Padrão)", "Estrategista de Vendas", "Auditor de Negócios", "Diretor Criativo"], value="Assessor Gold (Padrão)", show_label=False)
        net_box = gr.Checkbox(label="🌐 Conexão Web (Tempo Real)", value=False)
        
        gr.HTML("<div style='height: 40px;'></div>")
        
        gr.Markdown("### 🛡️ Cofre de Dados")
        btn_exportar = gr.Button("📥 Extrair Chat (Word)", variant="secondary")
        arq_exportado = gr.File(label="Protocolo", visible=False)
        
        btn_limpa = gr.Button("🧹 Resetar Cérebro Analítico", variant="secondary")
        msg_sys = gr.Textbox(show_label=False, interactive=False)
        btn_limpa.click(fn=limpar_banco_de_dados, outputs=msg_sys)
        
        btn_back = gr.Button("📦 Baixar Backup (ZIP)", variant="primary")
        msg_b = gr.Textbox(show_label=False)
        arq_b = gr.File(label="Arquivo ZIP", visible=False)
        btn_back.click(fn=gerar_backup, outputs=[arq_b, msg_b]).then(lambda: gr.update(visible=True), None, arq_b)

    with gr.Tabs():
        
        with gr.TabItem("🧠 O CÓDIGO DE OURO"):
            chat = gr.ChatInterface(
                fn=responder_chat_multimodal, 
                multimodal=True,
                additional_inputs=[persona_box, net_box],
                chatbot=gr.Chatbot(height="72vh", show_label=False, placeholder="SISTEMA DE ELITE ATIVO. INSERIR DIRETRIZ."),
                textbox=gr.MultimodalTextbox(placeholder="Comandos, arquivos ou links...", container=False, scale=7, show_label=False)
            )
            btn_exportar.click(fn=exportar_conversa, inputs=[chat.chatbot], outputs=[arq_exportado]).then(lambda: gr.update(visible=True), None, arq_exportado)

        with gr.TabItem("📑 AUDITORIA IA"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    arq_up = gr.File(label="Cofre de Documentos (PDF/XLSX)", file_count="multiple")
                    txt_ordem = gr.Textbox(label="Diretriz", lines=3, placeholder="O que eu devo auditar?")
                    with gr.Row():
                        c_img = gr.Checkbox(label="🖼️ Capa", value=False)
                        c_aud = gr.Checkbox(label="🔊 Áudio", value=False)
                        c_trib = gr.Checkbox(label="⚖️ Debate", value=False)
                    btn_exe = gr.Button("INICIAR AUDITORIA", variant="primary", size="lg")
                with gr.Column(scale=6):
                    out_tela = gr.Textbox(label="Painel de Visualização", lines=22, interactive=False)
                    with gr.Row():
                        out_word = gr.File(label="Dossiê Final")
                        out_aud = gr.Audio(label="Ouvir Síntese")
                    out_tel = gr.Textbox(show_label=False, lines=1, interactive=False)
            btn_exe.click(fn=gerar_dossie, inputs=[arq_up, txt_ordem, c_img, c_aud, c_trib], outputs=[msg_sys, out_word, out_aud, out_tela, out_tel])

        with gr.TabItem("🎬 ESTÚDIO GOLD"):
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37; margin-bottom: 15px;'>🖼️ FOTOGRAFIA DE LUXO</h3>")
                    img_sujeito = gr.Textbox(label="Objeto Principal", placeholder="Ex: Um frasco de perfume dourado...")
                    img_fundo = gr.Textbox(label="Atmosfera", placeholder="Luzes de estúdio...")
                    img_estilo = gr.Dropdown(choices=["Fotorrealista (8k)", "Cyberpunk", "Minimalista Escuro"], label="Estética", value="Fotorrealista (8k)")
                    btn_gerar_img = gr.Button("SINTETIZAR ARTE", variant="primary")
                    out_img = gr.Image(label="Arte", type="filepath")
                    btn_gerar_img.click(fn=gerar_imagem_estudio, inputs=[img_sujeito, img_fundo, img_estilo], outputs=[out_img])

                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37; margin-bottom: 15px;'>🎥 CINEMA IA</h3>")
                    vid_base = gr.Image(label="Base Visual (Opcional)", type="filepath")
                    vid_acao = gr.Textbox(label="Ação", placeholder="O perfume sendo pulverizado...")
                    vid_fundo = gr.Textbox(label="Cenário", placeholder="Fundo escuro...")
                    vid_mov = gr.Dropdown(choices=["Zoom In", "Zoom Out", "Pan Dir"], label="Câmera", value="Zoom In")
                    btn_gerar_vid = gr.Button("SINTETIZAR VÍDEO", variant="primary")
                    out_vid = gr.Video(label="Arquivo MP4")
                    msg_vid = gr.Textbox(show_label=False, interactive=False)
                    btn_gerar_vid.click(fn=gerar_video_ia, inputs=[vid_base, vid_acao, vid_fundo, vid_mov], outputs=[out_vid, msg_vid])
            
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37; margin-bottom: 15px;'>🎙️ SÍNTESE VOCAL NEURAL</h3>")
                    txt_aud = gr.Textbox(show_label=False, placeholder="Cole o roteiro...", lines=2)
                    btn_gerar_aud = gr.Button("GERAR LOCUÇÃO", variant="primary")
                    out_aud_estudio = gr.Audio(label="Arquivo MP3")
                    btn_gerar_aud.click(fn=falar_laudo_estudio, inputs=[txt_aud], outputs=[out_aud_estudio])

# ==========================================
# 8. ACESSOS NO COFRE
# ==========================================
lista_de_usuarios = []
for i in ["", "_1", "_2", "_3"]:
    u, s = os.environ.get(f"LOGIN_USUARIO{i}" if i=="" else f"USUARIO{i}"), os.environ.get(f"LOGIN_SENHA{i}" if i=="" else f"SENHA{i}")
    if u and s: lista_de_usuarios.append((u, s))

interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)), auth=lista_de_usuarios)
