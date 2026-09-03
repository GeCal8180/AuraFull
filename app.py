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

# --- GESTÃO DE SESSÕES ---
def listar_sessoes_chat():
    sessoes = [f.replace('.json', '') for f in os.listdir(DIR_CHATS) if f.endswith('.json')]
    sessoes.sort(reverse=True)
    return sessoes if sessoes else ["Nenhuma conversa salva"]

def carregar_sessao_chat(id_sessao):
    if not id_sessao or id_sessao == "Nenhuma conversa salva":
        return [], id_sessao
    caminho = f"{DIR_CHATS}/{id_sessao}.json"
    try:
        with open(caminho, "r", encoding="utf-8") as f: return json.load(f), id_sessao
    except: return [], id_sessao

def iniciar_novo_chat():
    novo_id = f"Projeto_{datetime.now().strftime('%d%m_%H%M%S')}"
    return [], novo_id, gr.update(choices=listar_sessoes_chat(), value=novo_id)

# ==========================================
# 3. EXTRAÇÃO MULTIMÍDIA E DADOS
# ==========================================
def encode_file_b64(caminho):
    with open(caminho, "rb") as f: return base64.b64encode(f.read()).decode('utf-8')

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
        elif nome.endswith(('.mp3', '.ogg', '.wav')):
            with open(caminho, "rb") as file: texto = f"[TRANSCRIÇÃO DE ÁUDIO]: {cliente_groq.audio.transcriptions.create(file=(caminho, file.read()), model='whisper-large-v3').text}\n"
        return texto
    except: return ""

# ==========================================
# 4. MOTORES HACKEADOS E VIP DE MÍDIA
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
            res = cliente_hf.image_to_image(
                image=caminho_imagem,
                prompt=prompt_edicao,
                model="timbrooks/instruct-pix2pix"
            )
            caminho_saida = f"{DIR_MIDIA}/Edit_{datetime.now().strftime('%H%M%S')}.jpg"
            res.save(caminho_saida)
            return caminho_saida
        except Exception as e:
            time.sleep(2)
            continue
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
            if imagem_base:
                cliente_i2v = Client("multimodalart/stable-video-diffusion", hf_token=chave_hf)
                return cliente_i2v.predict(imagem_base, api_name="/video")
            else:
                cliente_video = Client("multimodalart/zeroscope-v2", hf_token=chave_hf)
                return cliente_video.predict(prompt_cena[:150], api_name="/infer")
        except Exception:
            time.sleep(3 + tentativa)
            continue
    return None

# ==========================================
# 5. O CHAT AGÊNTICO ABSOLUTO
# ==========================================
def responder_chat_central(mensagem, historico, persona, usar_internet, id_sessao):
    texto_usuario = mensagem.get("text", "") if isinstance(mensagem, dict) else str(mensagem)
    arquivos = mensagem.get("files", []) if isinstance(mensagem, dict) else []
    
    contexto_extra = ""
    imagens_anexadas = []
    
    yield "⏳ *Sincronizando rede neural...*"
    
    for arq in arquivos:
        ext = arq.lower()
        if ext.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            imagens_anexadas.append(arq)
            yield f"👁️ *Escaneando visual da imagem enviada...*"
        else:
            yield f"📄 *Processando dados do documento...*"
            contexto_extra += f"\n[DOCUMENTO]:\n{extrair_texto(arq)}\n"
            
    if usar_internet and texto_usuario:
        yield "🌐 *Buscando atualizações de mercado na Web...*"
        try:
            resultados = DDGS().text(texto_usuario, max_results=4)
            contexto_extra += "\n\n[DADOS WEB]:\n" + "\n".join([f"Título: {r['title']} - Conteúdo: {r['body']}" for r in resultados])
        except: pass

    yield "🧠 *Construindo a solução executiva...*"

    sys_prompt = f"""Você é um {persona}, um Agente Centralizador Omnichannel de IA (nível Enterprise).
Responda com excelência técnica, tom direto e clareza.

SEUS 5 PODERES DE AÇÃO NO CHAT (Use apenas SE o usuário pedir explicitamente):
1. GERAR NOVA IMAGEM (Do zero): 
Inclua em uma linha isolada: [AÇÃO_IMAGEM: descrição altamente detalhada em inglês com 8k photorealistic | vertical] (Use 'vertical' para redes sociais ou 'quadrado').

2. EDITAR IMAGEM ENVIADA (Mudar fundo/estilo):
Se o usuário anexou uma imagem e quer modificá-la, inclua: [AÇÃO_EDITAR_IMAGEM: instrução direta da mudança em inglês]. Ex: [AÇÃO_EDITAR_IMAGEM: make the background a tropical beach]

3. GERAR VÍDEO (Novo ou Animar Foto):
Inclua em uma linha: [AÇÃO_VIDEO: curta descrição em inglês da ação com máx 25 palavras]

4. GERAR ÁUDIO / LOCUÇÃO:
Inclua em uma linha: [AÇÃO_AUDIO: texto exato em português a ser falado pela voz neural]

5. GERAR GRÁFICOS DE DADOS:
Para tabelas e planilhas numéricas, use obrigatoriamente blocos de código ```mermaid com gráficos visuais (pie chart ou bar chart)."""

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
            texto_visivel = re.sub(r'\[AÇÃO_\w+:.*?\]', '⚙️ *(Acionando motor multimídia...)*', resposta_acumulada)
            yield texto_visivel

    # ==========================================
    # PROCESSAMENTO DOS GATILHOS DE MÍDIA
    # ==========================================
    anexos_html = ""
    
    match_img = re.search(r'\[AÇÃO_IMAGEM:\s*(.*?)(?:\|\s*(\w+))?\]', resposta_acumulada)
    if match_img:
        prompt_i = match_img.group(1).strip()
        prop_i = "Vertical" if (match_img.group(2) and "vertical" in match_img.group(2).lower()) else "Quadrado"
        yield re.sub(r'\[AÇÃO_\w+:.*?\]', '', resposta_acumulada) + "\n\n🎨 *Fotografando cena em 8k (Aguarde)...*"
        cam_gerada = motor_gerar_imagem(prompt_i, prop_i)
        if cam_gerada and os.path.exists(cam_gerada):
            b64_img = encode_file_b64(cam_gerada)
            anexos_html += f"\n\n**🖼️ Imagem Gerada:**\n<img src='data:image/jpeg;base64,{b64_img}' style='max-width:100%; border-radius:18px; margin-top:8px; box-shadow:0 8px 20px rgba(0,0,0,0.15);' />\n"

    match_edit = re.search(r'\[AÇÃO_EDITAR_IMAGEM:\s*(.*?)\]', resposta_acumulada)
    if match_edit and imagens_anexadas:
        prompt_e = match_edit.group(1).strip()
        yield re.sub(r'\[AÇÃO_\w+:.*?\]', '', resposta_acumulada) + "\n\n🖌️ *Modificando inteligentemente a imagem enviada (Preservando objeto)...*"
        cam_edit = motor_editar_imagem(imagens_anexadas[-1], prompt_e)
        if cam_edit and os.path.exists(cam_edit):
            b64_img = encode_file_b64(cam_edit)
            anexos_html += f"\n\n**✨ Imagem Editada:**\n<img src='data:image/jpeg;base64,{b64_img}' style='max-width:100%; border-radius:18px; margin-top:8px; border: 2px solid #10A37F;' />\n"
        else:
            anexos_html += "\n\n*(⚠️ Ocorreu um erro ao editar a imagem nos servidores públicos. Tente novamente em alguns segundos.)*"

    match_aud = re.search(r'\[AÇÃO_AUDIO:\s*(.*?)\]', resposta_acumulada)
    if match_aud:
        texto_loc = match_aud.group(1).strip()
        yield re.sub(r'\[AÇÃO_\w+:.*?\]', '', resposta_acumulada) + "\n\n🎙️ *Gravando locução de estúdio...*"
        cam_aud = motor_gerar_audio(texto_loc)
        if cam_aud and os.path.exists(cam_aud):
            b64_aud = encode_file_b64(cam_aud)
            anexos_html += f"\n\n**🔊 Locução Pronta:**\n<audio controls src='data:audio/mp3;base64,{b64_aud}' style='width:100%; margin-top:8px;'></audio>\n"

    match_vid = re.search(r'\[AÇÃO_VIDEO:\s*(.*?)\]', resposta_acumulada)
    if match_vid:
        prompt_v = match_vid.group(1).strip()
        yield re.sub(r'\[AÇÃO_\w+:.*?\]', '', resposta_acumulada) + "\n\n🎬 *Conectando aos supercomputadores de vídeo (Fila VIP - Isso leva alguns instantes)...*"
        img_referencia = imagens_anexadas[-1] if imagens_anexadas else None
        cam_vid = motor_gerar_video(prompt_v, img_referencia)
        if cam_vid and os.path.exists(cam_vid):
            b64_vid = encode_file_b64(cam_vid)
            anexos_html += f"\n\n**🎥 Vídeo Gerado:**\n<video controls style='max-width:100%; border-radius:18px; margin-top:8px;' src='data:video/mp4;base64,{b64_vid}'></video>\n"

    resposta_final_limpa = re.sub(r'\[AÇÃO_\w+:.*?\]', '', resposta_acumulada).strip() + anexos_html
    yield resposta_final_limpa

    try:
        sessao_alvo = id_sessao if (id_sessao and id_sessao != "Nenhuma conversa salva") else f"Projeto_{datetime.now().strftime('%d%m_%H%M%S')}"
        arq_sessao = f"{DIR_CHATS}/{sessao_alvo}.json"
        historico_atual = []
        if os.path.exists(arq_sessao):
            with open(arq_sessao, "r", encoding="utf-8") as f: historico_atual = json.load(f)
        historico_atual.append({"role": "user", "content": texto_usuario if texto_usuario else "[Arquivo/Mídia Anexada]"})
        historico_atual.append({"role": "assistant", "content": resposta_final_limpa})
        with open(arq_sessao, "w", encoding="utf-8") as f:
            json.dump(historico_atual, f, ensure_ascii=False, indent=4)
    except: pass

def exportar_conversa_docx(historico):
    if not historico: return None
    pasta = f"{DIR_CASOS}/Chat_{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(pasta, exist_ok=True)
    cam_word = f"{pasta}/Historico_Projeto.docx"
    doc = docx.Document()
    doc.add_heading('Relatório da Sessão Titã', 0)
    for item in historico:
        if isinstance(item, dict):
            autor = "Você:" if item.get("role") == "user" else "Central de IA:"
            doc.add_heading(autor, level=2)
            doc.add_paragraph(re.sub(r'<.*?>', '', item.get("content", "")))
    doc.save(cam_word)
    return cam_word

def gerar_backup_zip():
    cam = "./Backup_Projetos_Central.zip"
    with zipfile.ZipFile(cam, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(DIRETORIO):
            if "Banco_de_Dados_Vetorial" not in root:
                for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), DIRETORIO))
    return cam

def gerar_dossie_lote(arquivos, instrucao, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Forneça uma instrução.", None, "", ""
    palavras = 0
    try:
        progresso(0.1, desc="Lendo arquivos...")
        pasta = f"{DIR_CASOS}/Lote_{datetime.now().strftime('%d_%m_%Y__%Hh%M')}"
        os.makedirs(pasta, exist_ok=True)
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for idx, arq in enumerate(arquivos):
                txt = extrair_texto(arq)
                palavras += len(txt.split())
                banco.add_texts([f"[FONTE: {os.path.basename(arq.name)}]\n{c}" for c in fatiador.split_text(txt)])
            
        progresso(0.5, desc="Cruzando inteligência...")
        contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        prompt = f"Você é um Especialista de Inteligência Sênior. DADOS:\n{contexto}\n\nINSTRUÇÃO: {instrucao}"
        resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ, max_tokens=4000).choices[0].message.content
        
        cam_word = f"{pasta}/Relatorio_Executivo.docx"
        doc = docx.Document()
        doc.add_heading('Relatório Executivo Oficial', 0)
        doc.add_paragraph(resposta)
        doc.save(cam_word)
        
        return "✅ Processamento Concluído!", cam_word, resposta, f"📊 {palavras} palavras analisadas com sucesso."
    except Exception as e:
        return f"Erro: {e}", None, "", ""

# ==========================================
# 6. DESIGN SYSTEM PWA NATIVO & MOBILE FIRST
# ==========================================
PWA_HEAD = """
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="mobile-web-app-capable" content="yes">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="theme-color" content="#18181B">
<title>Chat Titã</title>
"""

# CORREÇÃO APLICADA: gr.themes.sizes.radius_xxl 
tema_dola_premium = gr.themes.Soft(
    primary_hue="zinc", secondary_hue="slate", neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"], radius_size=gr.themes.sizes.radius_xxl,
).set(
    body_background_fill="#F4F4F5", body_background_fill_dark="#18181B",
    block_background_fill="#FFFFFF", block_background_fill_dark="#27272A",
    border_color_primary="transparent", border_color_primary_dark="transparent",
    block_border_width="0px", 
    button_primary_background_fill="#18181B", button_primary_background_fill_dark="#FAFAFA", 
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#000000"
)

css_dola_premium = """
footer {display: none !important;}
.gradio-container {max-width: 1200px !important; margin: auto !important;}
.tabs {border: none !important; background: transparent !important;}
.tab-nav {border-bottom: none !important; justify-content: center !important; font-size: 1.05em !important; margin-bottom: 25px; gap: 12px; padding-top: 15px;}
.tab-nav button {border-radius: 40px !important; border: 1px solid var(--border-color-primary) !important; padding: 12px 28px !important; background: var(--block-background-fill) !important; font-weight: 600; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);}
.tab-nav button.selected {background-color: var(--button-primary-background-fill) !important; color: var(--button-primary-text-color) !important; box-shadow: 0 10px 25px rgba(0,0,0,0.2);}
.box-painel {border-radius: 28px !important; padding: 30px !important; margin-bottom: 25px; box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.08); background: var(--block-background-fill);}
.dark .box-painel {box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.6);}
.chat-container {border-radius: 28px !important; overflow: hidden; box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.08);}
.message-wrap {border-radius: 28px !important;}
.message {border-radius: 24px !important; padding: 16px 24px !important; font-size: 16px !important; line-height: 1.6;}
::-webkit-scrollbar {width: 6px; height: 6px;}
::-webkit-scrollbar-track {background: transparent;}
::-webkit-scrollbar-thumb {background: #ccc; border-radius: 10px;}
"""

# ==========================================
# 7. CONSTRUÇÃO DA INTERFACE VISUAL
# ==========================================
with gr.Blocks(title="Chat Titã AI", theme=tema_dola_premium, css=css_dola_premium, head=PWA_HEAD) as interface:
    
    id_sessao_atual = gr.State(f"Projeto_{datetime.now().strftime('%d%m_%H%M%S')}")

    with gr.Sidebar(open=False):
        gr.Markdown("### 🗂️ Projetos e Sessões")
        btn_novo_chat = gr.Button("➕ Iniciar Novo Projeto", variant="primary")
        lista_chats = gr.Dropdown(choices=listar_sessoes_chat(), label="Histórico Salvo", interactive=True)
        btn_carregar = gr.Button("Continuar Projeto")
        btn_att_lista = gr.Button("🔄 Atualizar Sessões", variant="secondary")
        btn_att_lista.click(fn=lambda: gr.update(choices=listar_sessoes_chat()), outputs=[lista_chats])

    with gr.Tabs():
        
        # O ÚNICO CHAT QUE VOCÊ PRECISA USAR PARA TUDO
        with gr.TabItem("💬 Omnichannel Chat"):
            with gr.Accordion("⚙️ Especialidade e Dados", open=False):
                with gr.Row():
                    persona_box = gr.Dropdown(choices=["Assistente Universal", "Copywriter (Ads/TikTok)", "Auditor de Dados"], value="Assistente Universal", label="Especialidade", scale=3)
                    net_box = gr.Checkbox(label="🌐 Conectar à Web", value=False, scale=1)
                    btn_exportar = gr.Button("💾 Exportar Documento (Word)", variant="secondary", scale=1)
            
            chat = gr.ChatInterface(
                fn=responder_chat_central, multimodal=True, additional_inputs=[persona_box, net_box, id_sessao_atual],
                chatbot=gr.Chatbot(height=720, placeholder="Envie fotos para editar, crie vídeos e imagens do zero, analise dados... Tudo acontece por aqui!", bubble_full_width=False, render_markdown=True),
                textbox=gr.MultimodalTextbox(placeholder="Digite ou anexe arquivos aqui...", container=False, scale=7),
                submit_btn="Enviar 🚀", retry_btn="🔄 Refazer", undo_btn="✏️ Editar", clear_btn="🗑️ Limpar Conversa"
            )
            arq_exportado = gr.File(label="Arquivo DOCX", visible=False)
            btn_exportar.click(fn=exportar_conversa_docx, inputs=[chat.chatbot], outputs=[arq_exportado]).then(lambda: gr.update(visible=True), None, arq_exportado)
            
            btn_carregar.click(fn=carregar_sessao_chat, inputs=[lista_chats], outputs=[chat.chatbot, id_sessao_atual])
            btn_novo_chat.click(fn=iniciar_novo_chat, outputs=[chat.chatbot, id_sessao_atual, lista_chats])
            
        # ABA AUXILIAR DE EXPLORADOR
        with gr.TabItem("📑 Leitor de Lotes Pesados"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    arq_lote = gr.File(label="Lote de Documentos (PDF, Word, Excel)", file_count="multiple")
                    txt_instrucao = gr.Textbox(label="Instrução para a IA", lines=3, placeholder="O que você quer extrair ou cruzar destes documentos?")
                    btn_lote = gr.Button("Processar Lote Inteiro", variant="primary")
                    msg_status = gr.Textbox(show_label=False, interactive=False)
                with gr.Column(scale=6):
                    txt_relatorio = gr.Textbox(label="Relatório Gerado", lines=18, interactive=False)
                    out_doc_lote = gr.File(label="Baixar Relatório (Word)")
            btn_lote.click(fn=gerar_dossie_lote, inputs=[arq_lote, txt_instrucao], outputs=[msg_status, out_doc_lote, txt_relatorio, msg_status])

        # ABA AUXILIAR DE EXPLORADOR
        with gr.TabItem("🗂️ Galeria e Cofre"):
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### 🖼️ Álbum de Fotos Geradas e Editadas")
                    galeria_fotos = gr.Gallery(label="Galeria Visual", columns=4, height="auto")
                    btn_att_gal = gr.Button("🔄 Atualizar Álbum", variant="primary")
                    btn_att_gal.click(fn=atualizar_galeria_imagens, outputs=[galeria_fotos])
                
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### 📁 Explorador de Mídias (Vídeos, PDFs, Áudios)")
                    lista_geral = gr.File(label="Arquivos Baixáveis", file_count="multiple", interactive=False)
                    btn_att_arquivos = gr.Button("🔄 Atualizar Lista", variant="primary")
                    btn_att_arquivos.click(fn=listar_arquivos_mortos, outputs=[lista_geral])
                    btn_zip = gr.Button("Baixar Tudo de Uma Vez (ZIP)")
                    out_zip = gr.File(label="Download do ZIP", visible=False)
                    btn_zip.click(fn=gerar_backup_zip, outputs=[out_zip]).then(lambda: gr.update(visible=True), None, out_zip)

lista_usuarios = []
for i in ["", "_1", "_2", "_3"]:
    u = os.environ.get(f"LOGIN_USUARIO{i}" if i=="" else f"USUARIO{i}")
    s = os.environ.get(f"LOGIN_SENHA{i}" if i=="" else f"SENHA{i}")
    if u and s: lista_usuarios.append((u, s))

interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)), auth=lista_usuarios, theme=tema_dola_premium, css=css_dola_premium)
