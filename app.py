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
MODELO_GROQ = "llama-3.3-70b-versatile"
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

def listar_arquivos_mortos():
    arquivos = []
    for d in [DIR_CASOS, DIR_MIDIA]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                if f.endswith(('.docx', '.pdf', '.mp3', '.jpg', '.mp4')): arquivos.append(os.path.join(root, f))
    arquivos.sort(key=os.path.getmtime, reverse=True)
    return arquivos

# ==========================================
# 3. EXTRAÇÃO E LOGO BLINDADA (BASE64)
# ==========================================
def renderizar_logo():
    # Esta função injeta a imagem fisicamente no layout, impedindo o erro visual.
    caminho_imagem = "chamariz-sem-fundo.jpg"
    if os.path.exists(caminho_imagem):
        with open(caminho_imagem, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f'''
        <div style="text-align: center; margin-bottom: 15px;">
            <img src="data:image/jpeg;base64,{encoded}" style="max-width: 90%; margin: 0 auto; filter: drop-shadow(0px 10px 15px rgba(0,0,0,0.5));">
        </div>
        '''
    else:
        return '<h1 style="text-align: center; color: #D4AF37; font-family: sans-serif;">🏆 O Código de Ouro</h1>'

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
        elif nome.endswith(('.mp3', '.ogg', '.wav')):
            with open(caminho, "rb") as file: texto = f"[TRANSCRIÇÃO]: {cliente_groq.audio.transcriptions.create(file=(caminho, file.read()), model='whisper-large-v3').text}\n"
        return texto
    except: return ""

def limpar_banco_de_dados():
    try:
        if os.path.exists(DIR_CHROMA): shutil.rmtree(DIR_CHROMA)
        return "🧹 Memória limpa com sucesso. O Código de Ouro esqueceu os PDFs anteriores."
    except Exception as e: return f"Erro: {e}"

# ==========================================
# 4. CHAT ULTRAMODERNO (CORE)
# ==========================================
def responder_chat_multimodal(mensagem, historico, persona, usar_internet):
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

    sys_prompt = f"Você atua no sistema 'O Código de Ouro' e é um {persona}. Responda com excelência absoluta, clareza e formatação impecável de alto nível."
    mensagens = [{"role": "system", "content": sys_prompt}]
    
    for item in historico:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if isinstance(content, (tuple, list, dict)): content = "[Mídia Anexada]"
            if content: mensagens.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            h_u, h_a = item
            if h_u: mensagens.append({"role": "user", "content": "[Mídia Anexada]" if isinstance(h_u, (tuple, list, dict)) else str(h_u)})
            if h_a: mensagens.append({"role": "assistant", "content": str(h_a)})
    
    texto_final = texto_usuario + contexto_extra
    if imagens and not texto_final.strip(): texto_final = "Analise esta imagem em detalhes absolutos."
    elif not imagens and not texto_final.strip(): return "⚠️ Por favor, insira um comando."

    if imagens:
        conteudo_msg = [{"type": "text", "text": texto_final}]
        for img in imagens: conteudo_msg.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"}})
        mensagens.append({"role": "user", "content": conteudo_msg})
        modelo = MODELO_VISAO
    else:
        mensagens.append({"role": "user", "content": texto_final})
        modelo = MODELO_GROQ

    return cliente_groq.chat.completions.create(messages=mensagens, model=modelo, max_tokens=3000).choices[0].message.content

def exportar_conversa(historico):
    if not historico: return None
    pasta = f"{DIR_CASOS}/Chat_{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(pasta, exist_ok=True)
    cam_word = f"{pasta}/Sessao_O_Codigo_De_Ouro.docx"
    doc = docx.Document()
    doc.add_heading('Sessão - O Código de Ouro', 0)
    for item in historico:
        if isinstance(item, dict):
            role = "Você:" if item.get("role") == "user" else "O Código de Ouro:"
            content = "[Arquivo]" if isinstance(item.get("content"), (tuple, list, dict)) else str(item.get("content"))
            doc.add_heading(role, level=2)
            doc.add_paragraph(content)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            h_u, h_a = item
            doc.add_heading('Você:', level=2)
            doc.add_paragraph("[Arquivo]" if isinstance(h_u, (tuple, list, dict)) else str(h_u))
            doc.add_heading('O Código de Ouro:', level=2)
            doc.add_paragraph(str(h_a))
    doc.save(cam_word)
    return cam_word

# ==========================================
# 5. DOSSIÊ E MESA DO DIRETOR (TRADUÇÃO)
# ==========================================
def gerar_dossie(arquivos, instrucao, usar_img, usar_aud, usar_tribunal, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Faltou instrução", None, None, "", ""
    t_inicio = time.time()
    palavras = 0
    try:
        progresso(0.1, desc="Processando Ouro...")
        pasta = f"{DIR_CASOS}/Projeto_{datetime.now().strftime('%d_%m_%Y__%Hh%M')}"
        os.makedirs(pasta, exist_ok=True)
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for idx, arq in enumerate(arquivos):
                progresso((0.1 + (0.3 * (idx/len(arquivos)))), desc="Lendo base de dados...")
                txt = extrair_texto(arq)
                palavras += len(txt.split())
                banco.add_texts([f"[FONTE: {os.path.basename(arq.name)}]\n{c}" for c in fatiador.split_text(txt)])
            
        progresso(0.5, desc="Cruzando inteligência...")
        contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        regra_imagem = "\nNo final, pule uma linha e escreva: [IMAGEM: descreva em INGLÊS uma cena fotorrealista para este conteúdo]" if usar_img else ""
        prompt = f"Você é o analista sênior de 'O Código de Ouro'. Cite as fontes. {regra_imagem}\nDADOS: {contexto}\nAÇÃO: {instrucao}"
        resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ, max_tokens=4000).choices[0].message.content
        
        cam_img = None
        if usar_img:
            progresso(0.8, desc="Forjando Arte...")
            match = re.search(r'\[IMAGEM:\s*(.*?)\]', resposta, re.IGNORECASE)
            resposta_limpa = re.sub(r'\[IMAGEM:\s*(.*?)\]', '', resposta, flags=re.IGNORECASE).strip()
            prompt_img = match.group(1).strip() if match else "Minimalist corporate presentation cover, pure gold accents"
            cam_img = f"{pasta}/Capa_Projeto.jpg"
            cliente_hf.text_to_image(prompt_img, model="black-forest-labs/FLUX.1-schnell").save(cam_img)
        else: resposta_limpa = resposta

        progresso(0.9, desc="Exportando Ouro...")
        cam_word = f"{pasta}/Auditoria_Codigo_de_Ouro.docx"
        doc = docx.Document()
        doc.add_heading('Relatório - O Código de Ouro', 0)
        if cam_img: doc.add_picture(cam_img, width=Inches(6.0))
        doc.add_paragraph(resposta_limpa)
        doc.save(cam_word)
        
        progresso(1.0, desc="Concluído!")
        cam_audio = f"{pasta}/Voz_Ouro.mp3"
        if usar_aud:
            with open(f"{pasta}/temp.txt", "w", encoding="utf-8") as f: f.write(resposta_limpa[:3000].replace('*', ''))
            os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{pasta}/temp.txt" --write-media "{cam_audio}"')
        return "✅ Análise Gold Concluída", cam_word, cam_audio if usar_aud else None, resposta_limpa, f"📊 STATUS: {palavras} palavras auditadas."
    except Exception as e: return f"Erro: {e}", None, None, "", ""

def aprimorar_prompt(sujeito, fundo, estilo, tipo="imagem"):
    if not sujeito: return None
    instrucao = f"Traduza fielmente para INGLÊS. Sujeito: {sujeito} | Fundo: {fundo} | Estilo: {estilo}. Adicione termos de altíssima qualidade visual, 8k, photorealistic. Responda APENAS o texto em inglês." if tipo=="imagem" else f"Traduza para INGLÊS (MÁXIMO 40 PALAVRAS). Ação: {sujeito} | Fundo: {fundo} | Movimento: {estilo}. Responda APENAS o texto em inglês."
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
        try: return Client("multimodalart/stable-video-diffusion").predict(imagem_base, api_name="/video"), "✅ Imagem Gold animada!"
        except Exception: return None, "⚠️ Servidores lotados. Tente novamente."
    else:
        if not sujeito: return None, "⚠️ Preencha a Ação ou envie uma Imagem Base."
        try: return Client("multimodalart/zeroscope-v2").predict(aprimorar_prompt(sujeito, fundo, movimento, "vídeo"), api_name="/infer"), "✅ Vídeo Gold gerado!"
        except Exception: return None, "⚠️ Servidores lotados. Tente novamente."

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
    return cam, "📦 Cofre de Ouro Pronto para Download"

# ==========================================
# 7. DESIGN SYSTEM V10.0 (O CÓDIGO DE OURO)
# ==========================================
tema_ultra = gr.themes.Soft(
    primary_hue="amber", secondary_hue="zinc", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"], radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#F3F4F6", body_background_fill_dark="#0F172A",
    block_background_fill="#FFFFFF", block_background_fill_dark="#1E293B",
    border_color_primary="#E5E7EB", border_color_primary_dark="#334155",
    block_border_width="1px", 
    button_primary_background_fill="#D4AF37", # Dourado de Ouro Real
    button_primary_background_fill_dark="#C89B3C", 
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#F3F4F6",
    button_secondary_background_fill_dark="#334155"
)

css_ultra = """
/* Esconder rodapé do sistema */
footer {display: none !important;} 
.gradio-container {max-width: 1400px !important; padding-top: 0 !important; font-family: 'Inter', sans-serif;} 

/* Estilização Premium das Abas */
.tabs {border: none !important; background: transparent !important;} 
.tab-nav {border-bottom: 2px solid #E5E7EB !important; padding: 10px 20px 0 !important; background-color: transparent !important;} 
.dark .tab-nav {border-color: #334155 !important;}
.tab-nav button {font-size: 1.1em !important; font-weight: 600 !important; border-radius: 12px 12px 0 0 !important; padding: 12px 25px !important; color: #6B7280 !important; transition: all 0.3s ease;}
.tab-nav button.selected {color: #D4AF37 !important; border-bottom: 3px solid #D4AF37 !important; background: transparent !important;}

/* Painéis de Conteúdo com Efeito Vidro e Sombra */
.box-painel {border-radius: 16px; padding: 25px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05); border: 1px solid var(--border-color-primary); margin-bottom: 15px; transition: all 0.3s ease;} 
.dark .box-painel {box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);} 

/* Botões Primários Dourados */
button.primary {background: linear-gradient(135deg, #D4AF37 0%, #AA7C11 100%) !important; border: none !important; box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important; transition: transform 0.2s ease;}
button.primary:hover {transform: translateY(-2px); box-shadow: 0 6px 20px rgba(212, 175, 55, 0.5) !important;}

/* Chat Centralizado */
.chat-container {border-radius: 16px !important; border: 1px solid var(--border-color-primary) !important;}
"""

with gr.Blocks(title="O Código de Ouro", theme=tema_ultra, css=css_ultra, fill_width=True) as interface:
    
    # === MENU LATERAL (SIDEBAR DE ELITE) ===
    with gr.Sidebar(open=True):
        # Injeção da Logo Fisicamente via HTML e Base64 (Adeus Imagem Quebrada)
        gr.HTML(renderizar_logo())
        
        gr.HTML("<h3 style='text-align: center; color: #6B7280; font-weight: 400; margin-top: -10px; margin-bottom: 20px;'>Inteligência Estratégica</h3>")
        gr.HTML("<hr style='border: none; height: 1px; background-color: #E5E7EB; margin-bottom: 15px;'>")
        
        gr.Markdown("### ⚙️ Motor Analítico")
        persona_box = gr.Dropdown(choices=["Assessor Gold (Padrão)", "Copywriter Estratégico (Vendas)", "Consultor de Negócios", "Auditor de Dados", "Diretor de Arte (Design)"], value="Assessor Gold (Padrão)", label="Especialidade", show_label=False)
        net_box = gr.Checkbox(label="🌐 Acesso Web em Tempo Real", value=False)
        
        gr.HTML("<hr style='border: none; height: 1px; background-color: #E5E7EB; margin: 20px 0;'>")
        
        gr.Markdown("### 💾 Ações de Cofre")
        btn_exportar = gr.Button("📥 Exportar Sessão (Word)", variant="secondary")
        arq_exportado = gr.File(label="Histórico", visible=False)
        
        btn_limpa = gr.Button("🧹 Resetar Memória Analítica", variant="stop")
        msg_sys = gr.Textbox(show_label=False, interactive=False)
        btn_limpa.click(fn=limpar_banco_de_dados, outputs=msg_sys)
        
        btn_back = gr.Button("📦 Download do Cofre (Backup)", variant="primary")
        msg_b = gr.Textbox(show_label=False)
        arq_b = gr.File(label="Download ZIP", visible=False)
        btn_back.click(fn=gerar_backup, outputs=[arq_b, msg_b]).then(lambda: gr.update(visible=True), None, arq_b)

    # === ÁREA CENTRAL (SISTEMA) ===
    with gr.Tabs():
        
        # ABA 1: CHAT PRINCIPAL
        with gr.TabItem("💬 O Código de Ouro"):
            chat = gr.ChatInterface(
                fn=responder_chat_multimodal, 
                multimodal=True,
                additional_inputs=[persona_box, net_box],
                chatbot=gr.Chatbot(height="68vh", show_label=False, placeholder="Bem-vindo ao O Código de Ouro. O que vamos criar hoje?"),
                textbox=gr.MultimodalTextbox(placeholder="Digite sua estratégia, cole links, ou anexe documentos e clique na seta para enviar...", container=False, scale=7, show_label=False)
            )
            btn_exportar.click(fn=exportar_conversa, inputs=[chat.chatbot], outputs=[arq_exportado]).then(lambda: gr.update(visible=True), None, arq_exportado)

        # ABA 2: LEITURA DE LOTE
        with gr.TabItem("📑 Auditoria de Lote (PDFs)"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    arq_up = gr.File(label="Cofre de Documentos (Arraste PDFs)", file_count="multiple")
                    txt_ordem = gr.Textbox(label="Diretriz de Análise", lines=3, placeholder="Ex: Crie um resumo executivo deste lote...")
                    with gr.Row():
                        c_img = gr.Checkbox(label="🖼️ Capa Gold", value=False)
                        c_aud = gr.Checkbox(label="🔊 Narrador Gold", value=False)
                        c_trib = gr.Checkbox(label="⚖️ Debate", value=False)
                    btn_exe = gr.Button("Processar Inteligência", variant="primary", size="lg")
                with gr.Column(scale=6):
                    out_tela = gr.Textbox(label="Visualização do Relatório Gold", lines=22, interactive=False)
                    with gr.Row():
                        out_word = gr.File(label="Baixar Relatório Final")
                        out_aud = gr.Audio(label="Ouvir Síntese")
                    out_tel = gr.Textbox(show_label=False, lines=1, interactive=False)
            btn_exe.click(fn=gerar_dossie, inputs=[arq_up, txt_ordem, c_img, c_aud, c_trib], outputs=[msg_sys, out_word, out_aud, out_tela, out_tel])

        # ABA 3: MESA DO DIRETOR (MÍDIAS)
        with gr.TabItem("🎬 Estúdio Gold (Mídias)"):
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### 🖼️ Estúdio Fotográfico (FLUX.1)")
                    img_sujeito = gr.Textbox(label="Foco Principal (Sujeito)", placeholder="Ex: Tênis esportivo vermelho, textura premium...")
                    img_fundo = gr.Textbox(label="Cenário", placeholder="Ex: Fundo neutro cinza, iluminação de estúdio...")
                    img_estilo = gr.Dropdown(choices=["Fotorrealista (Cinematic 8k)", "Ilustração 3D (Pixar/Disney)", "Cyberpunk Neon", "Minimalista Clean"], label="Estilo Visual", value="Fotorrealista (Cinematic 8k)")
                    btn_gerar_img = gr.Button("Gerar Arte Gold", variant="primary")
                    out_img = gr.Image(label="Arte Final", type="filepath")
                    btn_gerar_img.click(fn=gerar_imagem_estudio, inputs=[img_sujeito, img_fundo, img_estilo], outputs=[out_img])

                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### 🎥 Set Cinematográfico (MP4)")
                    vid_base = gr.Image(label="Anexar Imagem para Animar (Opcional)", type="filepath")
                    vid_acao = gr.Textbox(label="Ação / Cena", placeholder="Ex: O tênis flutuando no ar em câmera lenta...")
                    vid_fundo = gr.Textbox(label="Cenário", placeholder="Ex: Fundo escuro com luzes de neon...")
                    vid_mov = gr.Dropdown(choices=["Zoom In Lento (Aproximar)", "Câmera Fixa", "Pan para Direita", "Estilo Drone"], label="Movimento", value="Zoom In Lento (Aproximar)")
                    btn_gerar_vid = gr.Button("Sintetizar Cena Gold", variant="primary")
                    out_vid = gr.Video(label="Vídeo Final")
                    msg_vid = gr.Textbox(show_label=False, interactive=False)
                    btn_gerar_vid.click(fn=gerar_video_ia, inputs=[vid_base, vid_acao, vid_fundo, vid_mov], outputs=[out_vid, msg_vid])
            
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### 🎙️ Sala de Dublagem")
                    txt_aud = gr.Textbox(show_label=False, placeholder="Cole o roteiro em português do seu anúncio aqui...", lines=2)
                    btn_gerar_aud = gr.Button("Sintetizar Voz Gold", variant="primary")
                    out_aud_estudio = gr.Audio(label="Áudio Pronto")
                    btn_gerar_aud.click(fn=falar_laudo_estudio, inputs=[txt_aud], outputs=[out_aud_estudio])

# ==========================================
# 8. ACESSOS NO COFRE
# ==========================================
lista_de_usuarios = []
for i in ["", "_1", "_2", "_3"]:
    u, s = os.environ.get(f"LOGIN_USUARIO{i}" if i=="" else f"USUARIO{i}"), os.environ.get(f"LOGIN_SENHA{i}" if i=="" else f"SENHA{i}")
    if u and s: lista_de_usuarios.append((u, s))

interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)), auth=lista_de_usuarios)
