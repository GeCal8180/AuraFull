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
import sqlite3
import requests
from datetime import datetime
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS

# ==========================================
# 1. CHAVES MESTRES E CONEXÕES
# ==========================================
chave_groq = os.environ.get("GROQ_API_KEY")
chave_hf = os.environ.get("HF_TOKEN")
chave_openrouter = os.environ.get("OPENROUTER_API_KEY") # O Passaporte para o Gigante

cliente_groq = Groq(api_key=chave_groq)
cliente_hf = InferenceClient(token=chave_hf)

# MOTOR DE CHOQUE (Velocidade e Visão via Groq)
MODELO_GROQ_RAPIDO = "llama-3.1-8b-instant"
MODELO_VISAO = "llama-3.2-90b-vision-preview"

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=chave_hf, 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# 2. DIRETÓRIOS E BANCO DE DADOS
# ==========================================
DIRETORIO = "./Central_IA_Master"
DIR_CHROMA = f"{DIRETORIO}/Banco_de_Dados_Vetorial"
DIR_CASOS = f"{DIRETORIO}/Projetos_Salvos"
DIR_MIDIA = f"{DIRETORIO}/Midia_Criada"
DB_PATH = f"{DIRETORIO}/codigo_de_ouro.db"

for d in [DIRETORIO, DIR_CASOS, DIR_MIDIA]:
    os.makedirs(d, exist_ok=True)

def iniciar_banco():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS memoria_sessoes (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT, conteudo TEXT, tipo TEXT)''')
    conn.commit()
    conn.close()

iniciar_banco()

def salvar_na_memoria(conteudo, tipo="Geral"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO memoria_sessoes (data, conteudo, tipo) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(conteudo), tipo))
    conn.commit()
    conn.close()

# ==========================================
# 3. RENDERIZAÇÃO DA LOGO E UTILS
# ==========================================
def renderizar_logo():
    caminhos = ["chamariz-sem-fundo.jpg", "./chamariz-sem-fundo.jpg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chamariz-sem-fundo.jpg")]
    caminho_real = next((c for c in caminhos if os.path.exists(c)), None)
    
    if caminho_real:
        with open(caminho_real, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f'''<div style="display: flex; justify-content: center; align-items: center; padding: 10px 0 25px 0;"><img src="data:image/jpeg;base64,{encoded}" style="max-width: 80%; filter: drop-shadow(0px 8px 15px rgba(212, 175, 55, 0.5));"></div>'''
    else:
        return '''<div style="text-align: center; margin-bottom: 25px; padding: 20px 10px; border-radius: 15px; background: linear-gradient(145deg, #BF953F, #B38728); box-shadow: 0 10px 25px rgba(212,175,55,0.3);"><h1 style="color: #000; font-family: 'Montserrat', sans-serif; font-weight: 900; letter-spacing: 2px; margin: 0; font-size: 18px;">O CÓDIGO DE OURO</h1></div>'''

def encode_image(image_path):
    with open(image_path, "rb") as image_file: return base64.b64encode(image_file.read()).decode('utf-8')

def extrair_texto(arquivo):
    caminho = arquivo.name if hasattr(arquivo, 'name') else arquivo
    nome = caminho.lower()
    texto = ""
    try:
        if nome.endswith('.pdf'):
            with pdfplumber.open(caminho) as pdf: texto += "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        elif nome.endswith(('.xlsx', '.csv')): texto = (pd.read_excel(caminho) if nome.endswith('.xlsx') else pd.read_csv(caminho)).to_string() 
        elif nome.endswith('.docx'): texto += "\n".join([p.text for p in docx.Document(caminho).paragraphs])
    except: pass
    return texto

def limpar_banco_de_dados():
    try:
        if os.path.exists(DIR_CHROMA): shutil.rmtree(DIR_CHROMA)
        return "🧹 Memória Neural Reiniciada."
    except Exception as e: return f"Erro: {e}"

# ==========================================
# 4. CONECTOR DO CÉREBRO GIGANTE (OPENROUTER FREE)
# ==========================================
def chamar_gigante_openrouter(mensagens):
    if not chave_openrouter: return None
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {chave_openrouter}", "Content-Type": "application/json", "HTTP-Referer": "https://ocodigodeouro.com", "X-Title": "O Codigo de Ouro"}
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free", # O modelo de 70 Bilhões 100% Gratuito
        "messages": mensagens,
        "max_tokens": 4000
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=40)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
        return None
    except: return None

# ==========================================
# 5. CHAT HÍBRIDO E WEBHOOKS
# ==========================================
def disparar_webhook(url, texto_contexto):
    if not url: return "⚠️ Informe a URL do Webhook (Make/Zapier)."
    if not texto_contexto: return "⚠️ Não há dados recentes para enviar."
    try:
        r = requests.post(url, json={"sistema": "O Código de Ouro", "dados": texto_contexto, "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        return "✅ Transmissão bem-sucedida! Dados enviados ao sistema externo." if r.status_code == 200 else f"✅ Comando disparado (Status: {r.status_code})"
    except Exception as e: return f"⚠️ Falha de transmissão: {e}"

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
            try: contexto_extra += "\n\n[DADOS WEB]:\n" + "\n".join([f"Fonte: {r['title']} - Resumo: {r['body']}" for r in DDGS().text(texto_usuario, max_results=3)])
            except: pass

        sys_prompt = f"Você atua no sistema de elite 'O Código de Ouro' e é um {persona}. Responda com excelência absoluta, foco comercial e tom majestoso."
        texto_final = texto_usuario + contexto_extra
        
        if imagens and not texto_final.strip(): texto_final = "Analise esta imagem em detalhes absolutos."
        elif not imagens and not texto_final.strip(): return "⚠️ Operação cancelada. Insira um comando."

        # TENTATIVA 1: MODO GIGANTE (Se tiver chave do OpenRouter e não for imagem)
        if chave_openrouter and not imagens:
            mensagens_gigante = [{"role": "system", "content": sys_prompt}]
            for item in historico:
                if isinstance(item, dict) and item.get("content"):
                    mensagens_gigante.append({"role": item.get("role"), "content": str(item.get("content"))})
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    if item[0]: mensagens_gigante.append({"role": "user", "content": str(item[0])})
                    if item[1]: mensagens_gigante.append({"role": "assistant", "content": str(item[1])})
            mensagens_gigante.append({"role": "user", "content": texto_final})
            
            resposta = chamar_gigante_openrouter(mensagens_gigante)
            if resposta: 
                salvar_na_memoria(f"USER: {texto_usuario}\nIA: {resposta}", "Chat (Modo Gigante)")
                return resposta

        # TENTATIVA 2: MODO RÁPIDO/VISÃO (Fallback perfeito e indestrutível)
        mensagens_groq = [{"role": "system", "content": sys_prompt}]
        for item in historico:
            if isinstance(item, dict) and item.get("content"): mensagens_groq.append({"role": item.get("role"), "content": str(item.get("content"))})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                if item[0]: mensagens_groq.append({"role": "user", "content": str(item[0])})
                if item[1]: mensagens_groq.append({"role": "assistant", "content": str(item[1])})
                
        if imagens:
            conteudo_msg = [{"type": "text", "text": texto_final}]
            for img in imagens: conteudo_msg.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"}})
            mensagens_groq.append({"role": "user", "content": conteudo_msg})
            modelo = MODELO_VISAO
        else:
            mensagens_groq.append({"role": "user", "content": texto_final})
            modelo = MODELO_GROQ_RAPIDO

        resposta = cliente_groq.chat.completions.create(messages=mensagens_groq, model=modelo, max_tokens=3000).choices[0].message.content
        salvar_na_memoria(f"USER: {texto_usuario}\nIA: {resposta}", "Chat (Modo Rápido)")
        return resposta
        
    except Exception as e: return f"⚠️ **Falha Crítica do Motor.** Detalhe técnico: `{str(e)}`"

def exportar_conversa(historico):
    if not historico: return None
    pasta = f"{DIR_CASOS}/Chat_{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(pasta, exist_ok=True)
    cam_word = f"{pasta}/Protocolo_O_Codigo_de_Ouro.docx"
    doc = docx.Document()
    doc.add_heading('Protocolo - O Código de Ouro', 0)
    for item in historico:
        if isinstance(item, dict):
            doc.add_heading("Você:" if item.get("role") == "user" else "O Código de Ouro:", level=2)
            doc.add_paragraph(str(item.get("content")))
    doc.save(cam_word)
    return cam_word

# ==========================================
# 6. AGENTE MESTRE AUTÔNOMO (COM MODO GIGANTE)
# ==========================================
def executar_agente_mestre(objetivo, progresso=gr.Progress()):
    if not objetivo: return "⚠️ Defina um objetivo.", None
    try:
        progresso(0.2, desc="🔍 Agente 1: Mapeando Mercado...")
        try: contexto_web = "\n".join([f"- {r['body']}" for r in DDGS().text(objetivo, max_results=5)])
        except: contexto_web = "Busca web indisponível."
        
        progresso(0.5, desc="🧠 Agente 2: Forjando a Estratégia de Elite...")
        prompt_estrategia = f"Você é o Agente Mestre do Código de Ouro. Objetivo: '{objetivo}'. \nDados de mercado reais: {contexto_web} \nCrie uma estratégia letal, um roteiro prático e, NO FINAL, escreva [IMAGEM: descreva em INGLÊS o ativo visual comercial fotorrealista]."
        
        estrategia = chamar_gigante_openrouter([{"role": "user", "content": prompt_estrategia}]) if chave_openrouter else None
        if not estrategia:
            estrategia = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt_estrategia}], model=MODELO_GROQ_RAPIDO, max_tokens=3000).choices[0].message.content
        
        progresso(0.8, desc="🎨 Agente 3: Renderizando Ativos Visuais...")
        match = re.search(r'\[IMAGEM:\s*(.*?)\]', estrategia, re.IGNORECASE)
        resposta_limpa = re.sub(r'\[IMAGEM:\s*(.*?)\]', '', estrategia, flags=re.IGNORECASE).strip()
        
        cam_img = f"{DIR_MIDIA}/Ativo_Visual_{datetime.now().strftime('%H%M%S')}.jpg"
        cliente_hf.text_to_image(match.group(1).strip() if match else f"Luxurious presentation for {objetivo}, 8k", model="black-forest-labs/FLUX.1-schnell").save(cam_img)
        
        salvar_na_memoria(resposta_limpa, "Agente Autônomo")
        progresso(1.0, desc="✅ Missão Cumprida.")
        return resposta_limpa, cam_img
    except Exception as e: return f"⚠️ Erro no Agente: {str(e)}", None

# ==========================================
# 7. ESTÚDIO E DOSSIÊ DE AUDITORIA
# ==========================================
def gerar_dossie(arquivos, instrucao, usar_img, usar_aud, usar_tribunal, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Faltou instrução", None, None, "", ""
    palavras = 0
    try:
        progresso(0.1, desc="Auditando Cofre...")
        pasta = f"{DIR_CASOS}/Projeto_{datetime.now().strftime('%d_%m_%Y__%Hh%M')}"
        os.makedirs(pasta, exist_ok=True)
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for arq in arquivos:
                txt = extrair_texto(arq)
                palavras += len(txt.split())
                banco.add_texts([f"[FONTE: {os.path.basename(arq.name)}]\n{c}" for c in fatiador.split_text(txt)])
            
        progresso(0.5, desc="Sintetizando Ouro...")
        contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        prompt = f"Analise como sistema O Código de Ouro.\nDADOS: {contexto}\nAÇÃO: {instrucao}"
        
        resposta = chamar_gigante_openrouter([{"role": "user", "content": prompt}]) if chave_openrouter else None
        if not resposta: resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ_RAPIDO, max_tokens=4000).choices[0].message.content
        
        progresso(0.9, desc="Exportando...")
        cam_word = f"{pasta}/Relatorio_Codigo_de_Ouro.docx"
        doc = docx.Document()
        doc.add_heading('Relatório - O Código de Ouro', 0)
        doc.add_paragraph(resposta)
        doc.save(cam_word)
        salvar_na_memoria(resposta, "Auditoria")
        
        cam_audio = f"{pasta}/Audio_Codigo_Ouro.mp3"
        if usar_aud:
            with open(f"{pasta}/temp.txt", "w", encoding="utf-8") as f: f.write(resposta[:3000].replace('*', ''))
            os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{pasta}/temp.txt" --write-media "{cam_audio}"')
        return "✅ Auditoria Concluída", cam_word, cam_audio if usar_aud else None, resposta, f"📊 STATUS: {palavras} palavras auditadas."
    except Exception as e: return f"Erro crítico: {e}", None, None, "", ""

def aprimorar_prompt(sujeito, fundo, estilo):
    try: return cliente_groq.chat.completions.create(messages=[{"role": "user", "content": f"Traduza para INGLÊS. Adicione 8k, photorealistic. Responda APENAS o texto. Sujeito: {sujeito} | Fundo: {fundo} | Estilo: {estilo}"}], model=MODELO_GROQ_RAPIDO, temperature=0.1).choices[0].message.content.strip()
    except: return f"{sujeito}, {fundo}, {estilo}"

def gerar_imagem_estudio(sujeito, fundo, estilo):
    if not sujeito: return None
    c = f"{DIR_MIDIA}/Img_{datetime.now().strftime('%H%M%S')}.jpg"
    cliente_hf.text_to_image(aprimorar_prompt(sujeito, fundo, estilo), model="black-forest-labs/FLUX.1-schnell").save(c)
    return c

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
# 8. DESIGN SYSTEM ALTO CONTRASTE (O CÓDIGO DE OURO)
# ==========================================
tema_ultra = gr.themes.Base(font=[gr.themes.GoogleFont("Montserrat"), "sans-serif"]).set(
    body_background_fill="#050505", body_background_fill_dark="#050505", body_text_color="#FFFFFF", body_text_color_dark="#FFFFFF",
    background_fill_primary="#0A0A0A", background_fill_primary_dark="#0A0A0A", background_fill_secondary="#111111", background_fill_secondary_dark="#111111",
    border_color_primary="#333333", border_color_primary_dark="#333333",
    block_background_fill="#0A0A0A", block_background_fill_dark="#0A0A0A",
    block_label_text_color="#D4AF37", block_label_text_color_dark="#D4AF37",
    input_background_fill="#141414", input_background_fill_dark="#141414", input_border_color="#444444", input_border_color_dark="#444444",
    button_primary_background_fill="linear-gradient(145deg, #D4AF37, #AA7C11)", button_primary_background_fill_dark="linear-gradient(145deg, #D4AF37, #AA7C11)",
    button_primary_text_color="#000000", button_secondary_background_fill="#1A1A1A", button_secondary_text_color="#D4AF37"
)

css_ultra = """
body, .gradio-container { background-color: #050505 !important; color: #FFFFFF !important; font-family: 'Montserrat', sans-serif !important; }
footer { display: none !important; }
span, p, label, h1, h2, h3, h4, .markdown-text, .chatbot { color: #F3F4F6 !important; }
h3 { color: #D4AF37 !important; }
input:-webkit-autofill { -webkit-box-shadow: 0 0 0 30px #111111 inset !important; -webkit-text-fill-color: #FFFFFF !important; }
textarea, input, select, .wrap-inner, .dropdown-menu, .wrap { background-color: #111111 !important; color: #FFFFFF !important; border: 1px solid #333333 !important; border-radius: 12px !important; }
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

with gr.Blocks(title="O Código de Ouro", theme=tema_ultra, css=css_ultra, fill_width=True, js="function() { document.body.classList.add('dark'); }") as interface:
    
    with gr.Sidebar(open=True):
        gr.HTML(renderizar_logo())
        status_cerebro = "🔵 **Modo Gigante (70B+)**" if os.environ.get("OPENROUTER_API_KEY") else "🟢 **Modo Rápido (8B)**"
        gr.HTML("<h3 style='text-align: center; color: #D4AF37; text-transform: uppercase; font-size: 10px; letter-spacing: 3px; margin-bottom: 20px;'>Inteligência Suprema</h3>")
        
        with gr.Accordion("⚙️ Engine Operacional", open=False):
            gr.Markdown(f"{status_cerebro}")
            persona_box = gr.Dropdown(choices=["Assessor Gold (Padrão)", "Estrategista de Vendas", "Auditor de Negócios"], value="Assessor Gold (Padrão)", show_label=False)
            net_box = gr.Checkbox(label="🌐 Conexão Web", value=False)
            btn_limpa = gr.Button("🧹 Limpar Cache", variant="secondary")
            msg_sys = gr.Textbox(show_label=False, interactive=False)
            btn_limpa.click(fn=limpar_banco_de_dados, outputs=msg_sys)
        
        with gr.Accordion("📡 Automação (Webhooks)", open=False):
            gr.Markdown("*Integração com Make.com ou Zapier*")
            url_webhook = gr.Textbox(placeholder="Cole a URL do Webhook aqui...", show_label=False)
            texto_export = gr.Textbox(placeholder="Texto a enviar...", lines=2, show_label=False)
            btn_web = gr.Button("Transmitir Dados", variant="primary")
            msg_web = gr.Textbox(show_label=False, interactive=False)
            btn_web.click(fn=disparar_webhook, inputs=[url_webhook, texto_export], outputs=msg_web)

        gr.HTML("<hr style='border: none; border-bottom: 1px solid #222; margin: 20px 0;'>")
        btn_back = gr.Button("📦 Baixar Backup Geral", variant="primary")
        msg_b = gr.Textbox(show_label=False)
        arq_b = gr.File(label="Arquivo", visible=False)
        btn_back.click(fn=gerar_backup, outputs=[arq_b, msg_b]).then(lambda: gr.update(visible=True), None, arq_b)

    with gr.Tabs():
        
        with gr.TabItem("🧠 O CÓDIGO DE OURO"):
            chat = gr.ChatInterface(
                fn=responder_chat_multimodal, multimodal=True, additional_inputs=[persona_box, net_box],
                chatbot=gr.Chatbot(height="72vh", show_label=False, placeholder="SISTEMA ATIVO. QUAL A DIRETRIZ?"),
                textbox=gr.MultimodalTextbox(placeholder="Comandos, arquivos ou links...", container=False, scale=7, show_label=False)
            )

        with gr.TabItem("🤖 AGENTE MESTRE"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    gr.Markdown("### DELEGUE UMA MISSÃO\nO Agente pesquisa, cria a estratégia e gera imagens de forma autônoma.")
                    txt_missao = gr.Textbox(label="Missão", lines=4, placeholder="Ex: Crie o roteiro de um café premium...")
                    btn_agente = gr.Button("INICIAR PROTOCOLO", variant="primary", size="lg")
                with gr.Column(scale=6):
                    out_estrat = gr.Textbox(label="Estratégia Final", lines=15, interactive=False)
                    out_arte = gr.Image(label="Ativo Visual Gerado", type="filepath")
            btn_agente.click(fn=executar_agente_mestre, inputs=[txt_missao], outputs=[out_estrat, out_arte])

        with gr.TabItem("📑 AUDITORIA DE DADOS"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    arq_up = gr.File(label="Cofre de Documentos (PDF/XLS)", file_count="multiple")
                    txt_ordem = gr.Textbox(label="Diretriz de Análise", lines=3)
                    with gr.Row():
                        c_img = gr.Checkbox(label="🖼️ Capa", value=False)
                        c_aud = gr.Checkbox(label="🔊 Áudio", value=False)
                    btn_exe = gr.Button("INICIAR AUDITORIA", variant="primary")
                with gr.Column(scale=6):
                    out_tela = gr.Textbox(label="Dossiê", lines=20)
                    with gr.Row():
                        out_word = gr.File(label="Dossiê Final")
                        out_aud = gr.Audio(label="Síntese")
                    out_tel = gr.Textbox(show_label=False, lines=1)
            btn_exe.click(fn=gerar_dossie, inputs=[arq_up, txt_ordem, c_img, c_aud, gr.State(False)], outputs=[msg_sys, out_word, out_aud, out_tela, out_tel])

        with gr.TabItem("🎬 ESTÚDIO GOLD"):
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37;'>🖼️ FOTOGRAFIA</h3>")
                    img_sujeito = gr.Textbox(label="Objeto Principal", placeholder="Ex: Um frasco de perfume dourado...")
                    img_fundo = gr.Textbox(label="Atmosfera")
                    img_estilo = gr.Dropdown(choices=["Fotorrealista (8k)", "Cyberpunk"], label="Estética", value="Fotorrealista (8k)")
                    btn_gerar_img = gr.Button("SINTETIZAR ARTE", variant="primary")
                    out_img_est = gr.Image(label="Arte", type="filepath")
                    btn_gerar_img.click(fn=gerar_imagem_estudio, inputs=[img_sujeito, img_fundo, img_estilo], outputs=[out_img_est])
                
                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37;'>🎙️ SÍNTESE VOCAL NEURAL</h3>")
                    txt_aud = gr.Textbox(show_label=False, placeholder="Cole o roteiro...", lines=4)
                    btn_gerar_aud = gr.Button("GERAR LOCUÇÃO", variant="primary")
                    out_aud_estudio = gr.Audio(label="Arquivo MP3")
                    btn_gerar_aud.click(fn=falar_laudo_estudio, inputs=[txt_aud], outputs=[out_aud_estudio])

lista_de_usuarios = [(os.environ.get(f"USUARIO{i}"), os.environ.get(f"SENHA{i}")) for i in ["", "_1", "_2"] if os.environ.get(f"USUARIO{i}")]
interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)), auth=lista_de_usuarios if lista_de_usuarios else None)
