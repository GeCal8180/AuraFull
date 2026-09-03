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
# 3. EXTRAÇÃO E LOGO IMPONENTE
# ==========================================
def renderizar_logo():
    caminho = "chamariz-sem-fundo.jpg"
    if os.path.exists(caminho):
        with open(caminho, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f'''
        <div style="display: flex; justify-content: center; align-items: center; padding: 25px 0; border-bottom: 1px solid #1F1F1F; margin-bottom: 20px;">
            <img src="data:image/jpeg;base64,{encoded}" style="max-width: 90%; filter: drop-shadow(0px 0px 25px rgba(212, 175, 55, 0.35));">
        </div>
        '''
    else:
        return '''
        <div style="text-align: center; padding: 30px 0; border-bottom: 1px solid #1F1F1F; margin-bottom: 20px;">
            <h2 style="color: #D4AF37; font-family: 'Montserrat', sans-serif; letter-spacing: 3px; font-weight: 800;">O CÓDIGO DE OURO</h2>
            <p style="color: red; font-size: 10px;">[Logo não encontrada]</p>
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
        elif nome.endswith(('.mp3', '.ogg', '.wav')):
            with open(caminho, "rb") as file: texto = f"[TRANSCRIÇÃO]: {cliente_groq.audio.transcriptions.create(file=(caminho, file.read()), model='whisper-large-v3').text}\n"
        return texto
    except: return ""

def limpar_banco_de_dados():
    try:
        if os.path.exists(DIR_CHROMA): shutil.rmtree(DIR_CHROMA)
        return "🧹 Memória limpa com sucesso. O sistema esqueceu os dados antigos."
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

        sys_prompt = f"Você atua no sistema de elite 'O Código de Ouro' e é um {persona}. Responda com excelência absoluta, clareza e formatação de altíssimo nível."
        mensagens = [{"role": "system", "content": sys_prompt}]
        
        for item in historico:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "")
                if isinstance(content, (tuple, list, dict)): content = "[Mídia Anexada Anteriormente]"
                if content and str(content).strip(): mensagens.append({"role": role, "content": str(content)})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                h_u, h_a = item
                if h_u: mensagens.append({"role": "user", "content": "[Mídia Anexada Anteriormente]" if isinstance(h_u, (tuple, list, dict)) else str(h_u)})
                if h_a: mensagens.append({"role": "assistant", "content": str(h_a)})
        
        texto_final = texto_usuario + contexto_extra
        if imagens and not texto_final.strip(): texto_final = "Analise esta imagem em detalhes absolutos."
        elif not imagens and not texto_final.strip(): return "⚠️ Operação cancelada. Por favor, insira um comando ou anexe um arquivo."

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
        # A blindagem: Se qualquer coisa falhar, a IA reporta o erro na tela ao invés de quebrar o site
        return f"⚠️ **Falha na Conexão Neural.** O servidor da inteligência rejeitou o pacote de dados. Detalhe do erro técnico: `{str(e)}`\n\n*Por favor, clique em 'Limpar Chat' ou tente enviar a mensagem novamente.*"

def exportar_conversa(historico):
    if not historico: return None
    pasta = f"{DIR_CASOS}/Chat_{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(pasta, exist_ok=True)
    cam_word = f"{pasta}/Sessao_Codigo_de_Ouro.docx"
    doc = docx.Document()
    doc.add_heading('Sessão - O Código de Ouro', 0)
    for item in historico:
        if isinstance(item, dict):
            role = "Comando:" if item.get("role") == "user" else "O Código de Ouro:"
            content = "[Arquivo]" if isinstance(item.get("content"), (tuple, list, dict)) else str(item.get("content"))
            doc.add_heading(role, level=2)
            doc.add_paragraph(content)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            h_u, h_a = item
            doc.add_heading('Comando:', level=2)
            doc.add_paragraph("[Arquivo]" if isinstance(h_u, (tuple, list, dict)) else str(h_u))
            doc.add_heading('O Código de Ouro:', level=2)
            doc.add_paragraph(str(h_a))
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
    instrucao = f"Traduza para INGLÊS. Sujeito: {sujeito} | Fundo: {fundo} | Estilo: {estilo}. Adicione termos: 8k, highly detailed, photorealistic. Responda APENAS o texto em inglês." if tipo=="imagem" else f"Traduza para INGLÊS (MÁX 40 PALAVRAS). Ação: {sujeito} | Fundo: {fundo} | Movimento: {estilo}. Responda APENAS o texto em inglês."
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
        except Exception: return None, "⚠️ Motor gráfico congestionado."
    else:
        if not sujeito: return None, "⚠️ Preencha a Ação."
        try: return Client("multimodalart/zeroscope-v2").predict(aprimorar_prompt(sujeito, fundo, movimento, "vídeo"), api_name="/infer"), "✅ Vídeo renderizado!"
        except Exception: return None, "⚠️ Motor gráfico congestionado."

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
    return cam, "📦 Arquivos Criptografados e Prontos"

# ==========================================
# 7. DESIGN ABSOLUTE DARK GOLD (O CÓDIGO DE OURO)
# ==========================================
tema_ultra = gr.themes.Default(
    font=[gr.themes.GoogleFont("Montserrat"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0A0A0A", 
    body_background_fill_dark="#0A0A0A",
    block_background_fill="#141414", 
    block_background_fill_dark="#141414",
    border_color_primary="#262626", 
    border_color_primary_dark="#262626",
    button_primary_background_fill="#D4AF37", 
    button_primary_background_fill_dark="#D4AF37", 
    button_primary_text_color="#000000",
    button_secondary_background_fill="#1A1A1A",
    button_secondary_background_fill_dark="#1A1A1A"
)

css_ultra = """
/* Forçar Modo Escuro Absoluto e Remover Rodapé */
body { background-color: #0A0A0A !important; color: #FFFFFF !important; }
footer {display: none !important;} 
.gradio-container {max-width: 1600px !important; padding: 0 !important; font-family: 'Montserrat', sans-serif;} 

/* Sidebar de Luxo */
.sidebar { background: linear-gradient(180deg, #111111 0%, #050505 100%) !important; border-right: 1px solid #1F1F1F !important; padding: 20px !important; box-shadow: 5px 0 20px rgba(0,0,0,0.8); }

/* Customização das Abas Superiores */
.tabs { background: transparent !important; border: none !important; }
.tab-nav { border-bottom: 1px solid #1F1F1F !important; justify-content: center !important; margin-bottom: 25px !important; background: #0A0A0A !important; }
.tab-nav button { color: #666666 !important; text-transform: uppercase; letter-spacing: 2px; font-weight: 600 !important; border: none !important; padding: 15px 30px !important; transition: 0.3s; }
.tab-nav button.selected { color: #D4AF37 !important; border-bottom: 2px solid #D4AF37 !important; background: transparent !important; }
.tab-nav button:hover { color: #FFFFFF !important; }

/* Blocos de Vidro Flutuante */
.box-painel { background: rgba(20, 20, 20, 0.7) !important; backdrop-filter: blur(10px); border: 1px solid #262626 !important; border-radius: 12px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); margin-bottom: 20px; transition: transform 0.3s; }
.box-painel:hover { transform: translateY(-3px); border-color: #333333 !important; }

/* Elementos de Chat Imponentes */
.chat-container { border: 1px solid #1F1F1F !important; background: #0F0F0F !important; border-radius: 12px !important; }
.message.user { background: #1A1A1A !important; border: 1px solid #262626 !important; }
.message.bot { background: #111111 !important; border-left: 3px solid #D4AF37 !important; }

/* Botão de Elite */
button.primary { background: linear-gradient(135deg, #D4AF37 0%, #AA7C11 100%) !important; color: #000000 !important; font-weight: bold !important; letter-spacing: 1px; border: none !important; box-shadow: 0 4px 15px rgba(212, 175, 55, 0.2) !important; transition: all 0.3s !important; }
button.primary:hover { transform: scale(1.02); box-shadow: 0 6px 20px rgba(212, 175, 55, 0.5) !important; }
"""

# Injetor JS para Forçar o Layout a Nascer no Dark Mode
forca_dark_mode = """
function() {
    document.body.classList.add('dark');
}
"""

with gr.Blocks(title="O Código de Ouro", theme=tema_ultra, css=css_ultra, fill_width=True, js=forca_dark_mode) as interface:
    
    with gr.Sidebar(open=True):
        gr.HTML(renderizar_logo())
        
        gr.HTML("<h3 style='text-align: center; color: #888888; text-transform: uppercase; font-size: 11px; letter-spacing: 2px; margin-bottom: 25px;'>Sistema Operacional de Elite</h3>")
        
        gr.Markdown("### ⚙️ Engine Estratégica")
        persona_box = gr.Dropdown(choices=["Assessor Gold (Padrão)", "Copywriter Tático (Vendas)", "Consultor de Negócios", "Auditor Analítico", "Diretor Criativo"], value="Assessor Gold (Padrão)", show_label=False)
        net_box = gr.Checkbox(label="🌐 Conexão Web (Tempo Real)", value=False)
        
        gr.HTML("<hr style='border: none; border-bottom: 1px solid #1F1F1F; margin: 30px 0;'>")
        
        gr.Markdown("### 🛡️ Operações de Cofre")
        btn_exportar = gr.Button("📥 Baixar Dossiê (Word)", variant="secondary")
        arq_exportado = gr.File(label="Protocolo", visible=False)
        
        btn_limpa = gr.Button("🧹 Purgar Memória", variant="stop")
        msg_sys = gr.Textbox(show_label=False, interactive=False)
        btn_limpa.click(fn=limpar_banco_de_dados, outputs=msg_sys)
        
        btn_back = gr.Button("📦 Extrair Cofre Criptografado", variant="primary")
        msg_b = gr.Textbox(show_label=False)
        arq_b = gr.File(label="Arquivo ZIP", visible=False)
        btn_back.click(fn=gerar_backup, outputs=[arq_b, msg_b]).then(lambda: gr.update(visible=True), None, arq_b)

    with gr.Tabs():
        
        with gr.TabItem("🧠 O CÓDIGO DE OURO"):
            chat = gr.ChatInterface(
                fn=responder_chat_multimodal, 
                multimodal=True,
                additional_inputs=[persona_box, net_box],
                chatbot=gr.Chatbot(height="70vh", show_label=False, placeholder="SISTEMA ATIVO. O QUE VAMOS EXECUTAR HOJE?"),
                textbox=gr.MultimodalTextbox(placeholder="Insira o comando tático, arquivos ou imagens...", container=False, scale=7)
            )
            btn_exportar.click(fn=exportar_conversa, inputs=[chat.chatbot], outputs=[arq_exportado]).then(lambda: gr.update(visible=True), None, arq_exportado)

        with gr.TabItem("📑 AUDITORIA DE DADOS"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    arq_up = gr.File(label="Arquivos Físicos (PDF/XLSX)", file_count="multiple")
                    txt_ordem = gr.Textbox(label="Diretriz Executiva", lines=3, placeholder="Ex: Processe estes relatórios financeiros...")
                    with gr.Row():
                        c_img = gr.Checkbox(label="🖼️ Capa Ilustrativa", value=False)
                        c_aud = gr.Checkbox(label="🔊 Briefing em Áudio", value=False)
                        c_trib = gr.Checkbox(label="⚖️ Análise de Conflitos", value=False)
                    btn_exe = gr.Button("INICIAR AUDITORIA", variant="primary", size="lg")
                with gr.Column(scale=6):
                    out_tela = gr.Textbox(label="Visualização do Dossiê", lines=22, interactive=False)
                    with gr.Row():
                        out_word = gr.File(label="Baixar Relatório Final")
                        out_aud = gr.Audio(label="Ouvir Síntese")
                    out_tel = gr.Textbox(show_label=False, lines=1, interactive=False)
            btn_exe.click(fn=gerar_dossie, inputs=[arq_up, txt_ordem, c_img, c_aud, c_trib], outputs=[msg_sys, out_word, out_aud, out_tela, out_tel])

        with gr.TabItem("🎬 ESTÚDIO GOLD"):
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### 🖼️ FOTOGRAFIA NEURAL")
                    img_sujeito = gr.Textbox(label="Objeto Principal", placeholder="Ex: Um relógio de luxo em ouro maciço...")
                    img_fundo = gr.Textbox(label="Atmosfera", placeholder="Ex: Em uma vitrine iluminada...")
                    img_estilo = gr.Dropdown(choices=["Fotorrealista (8k)", "Cyberpunk", "Minimalista Escuro"], label="Estética", value="Fotorrealista (8k)")
                    btn_gerar_img = gr.Button("RENDERIZAR ARTE", variant="primary")
                    out_img = gr.Image(label="Arte Final", type="filepath")
                    btn_gerar_img.click(fn=gerar_imagem_estudio, inputs=[img_sujeito, img_fundo, img_estilo], outputs=[out_img])

                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### 🎥 CINEMA IA")
                    vid_base = gr.Image(label="Base Visual (Opcional)", type="filepath")
                    vid_acao = gr.Textbox(label="Ação", placeholder="Ex: O relógio brilhando sob a luz...")
                    vid_fundo = gr.Textbox(label="Cenário", placeholder="Fundo noturno...")
                    vid_mov = gr.Dropdown(choices=["Zoom In Lento", "Zoom Out", "Pan para Direita"], label="Câmera", value="Zoom In Lento")
                    btn_gerar_vid = gr.Button("RENDERIZAR VÍDEO", variant="primary")
                    out_vid = gr.Video(label="Arquivo MP4")
                    msg_vid = gr.Textbox(show_label=False, interactive=False)
                    btn_gerar_vid.click(fn=gerar_video_ia, inputs=[vid_base, vid_acao, vid_fundo, vid_mov], outputs=[out_vid, msg_vid])
            
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### 🎙️ SÍNTESE VOCAL")
                    txt_aud = gr.Textbox(show_label=False, placeholder="Cole o roteiro em português do anúncio...", lines=2)
                    btn_gerar_aud = gr.Button("RENDERIZAR VOZ", variant="primary")
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
