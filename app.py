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
import io
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
chave_openrouter = os.environ.get("OPENROUTER_API_KEY")

cliente_groq = Groq(api_key=chave_groq)
cliente_hf = InferenceClient(token=chave_hf)

# NOVA GERAÇÃO DE MOTORES GROQ (GPT-OSS 20B e QWEN 27B VISION)
MODELO_GROQ_RAPIDO = "openai/gpt-oss-20b"
MODELO_VISAO = "qwen/qwen3.6-27b"

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
def buscar_caminho_logo():
    for root, dirs, files in os.walk("."):
        for f in files:
            if "chamariz" in f.lower() and "fundo" in f.lower():
                return os.path.join(root, f)
    return None

def renderizar_logo():
    caminho = buscar_caminho_logo()
    if caminho:
        extensao = "png" if caminho.lower().endswith(".png") else "jpeg"
        with open(caminho, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f'''<div style="display: flex; justify-content: center; align-items: center; padding: 15px 0 30px 0;"><img src="data:image/{extensao};base64,{encoded}" style="max-width: 75%; filter: drop-shadow(0px 8px 20px rgba(212, 175, 55, 0.4));"></div>'''
    else:
        return '''<div style="text-align: center; margin-bottom: 30px; padding: 25px 10px; border-radius: 12px; background: linear-gradient(145deg, #BF953F, #B38728); box-shadow: 0 10px 25px rgba(212,175,55,0.2);"><h1 style="color: #000; font-family: 'Outfit', sans-serif; font-weight: 800; letter-spacing: 4px; margin: 0; font-size: 20px;">O CÓDIGO DE OURO</h1></div>'''

def renderizar_logo_login():
    caminho = buscar_caminho_logo()
    if caminho:
        extensao = "png" if caminho.lower().endswith(".png") else "jpeg"
        with open(caminho, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        return f'''<div style="text-align: center; padding: 10px 0 20px 0;"><img src="data:image/{extensao};base64,{encoded}" style="max-width: 250px; filter: drop-shadow(0px 8px 20px rgba(212, 175, 55, 0.5)); margin-bottom: 15px;"><br><span style="color: #D4AF37; font-family: 'Outfit', sans-serif; letter-spacing: 4px; font-size: 12px; font-weight: 600; text-transform: uppercase;">Acesso Restrito</span></div>'''
    else:
        return '''<div style="text-align: center; padding: 10px 0 20px 0;"><h1 style="color: #D4AF37; font-family: 'Outfit', sans-serif; letter-spacing: 4px; font-size: 22px;">O CÓDIGO DE OURO</h1><br><span style="color: #888; font-family: 'Inter', sans-serif; letter-spacing: 2px; font-size: 11px;">ACESSO RESTRITO</span></div>'''

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
                    txt_digital = p.extract_text()
                    if txt_digital and len(txt_digital.strip()) > 30:
                        texto += txt_digital + "\n"
                    else:
                        try:
                            img = p.to_image(resolution=150).original
                            buffer = io.BytesIO()
                            img.save(buffer, format="JPEG")
                            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            msg = [{"role": "user", "content": [{"type": "text", "text": "Extract all readable text from this document image."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}]
                            resp = cliente_groq.chat.completions.create(model=MODELO_VISAO, messages=msg, max_tokens=1024, temperature=0.1)
                            texto += resp.choices[0].message.content + "\n"
                            time.sleep(0.5)
                        except: pass
        elif nome.endswith(('.xlsx', '.csv')): texto = (pd.read_excel(caminho) if nome.endswith('.xlsx') else pd.read_csv(caminho)).to_string() 
        elif nome.endswith('.docx'): texto += "\n".join([p.text for p in docx.Document(caminho).paragraphs])
        elif nome.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            try:
                img_b64 = encode_image(caminho)
                msg = [{"role": "user", "content": [{"type": "text", "text": "Extraia dados detalhados desta imagem."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}]
                resp = cliente_groq.chat.completions.create(model=MODELO_VISAO, messages=msg, max_tokens=1500, temperature=0.1)
                texto += f"[IMAGEM: {os.path.basename(caminho)}]\n" + resp.choices[0].message.content + "\n"
            except: pass
    except: pass
    return texto

def limpar_banco_de_dados():
    try:
        if os.path.exists(DIR_CHROMA): shutil.rmtree(DIR_CHROMA)
        return "🧹 Memória Neural Reiniciada."
    except Exception as e: return f"Erro: {e}"

# ==========================================
# 5. CHAT E WEBHOOKS
# ==========================================
def chamar_gigante_openrouter(mensagens):
    if not chave_openrouter: return None
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {chave_openrouter}", "Content-Type": "application/json", "HTTP-Referer": "https://ocodigodeouro.com", "X-Title": "O Codigo de Ouro"}
    payload = {"model": "meta-llama/llama-3.3-70b-instruct:free", "messages": mensagens, "max_tokens": 4000}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=40)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
        return None
    except: return None

def disparar_webhook(url, texto_contexto):
    if not url: return "⚠️ Informe a URL do Webhook."
    if not texto_contexto: return "⚠️ Não há dados recentes."
    try:
        r = requests.post(url, json={"sistema": "O Código de Ouro", "dados": texto_contexto, "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        return "✅ Transmissão bem-sucedida!" if r.status_code == 200 else f"✅ Comando disparado (Status: {r.status_code})"
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

        sys_prompt = f"Você atua no sistema de elite 'O Código de Ouro' e é um {persona}. Responda com excelência absoluta, foco em performance comercial e tom majestoso."
        texto_final = texto_usuario + contexto_extra
        
        if imagens and not texto_final.strip(): texto_final = "Analise esta imagem em detalhes absolutos."
        elif not imagens and not texto_final.strip(): return "⚠️ Insira um comando."

        if chave_openrouter and not imagens:
            mensagens_gigante = [{"role": "system", "content": sys_prompt}]
            for item in historico:
                if isinstance(item, dict) and item.get("content"): mensagens_gigante.append({"role": item.get("role"), "content": str(item.get("content"))})
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    if item[0]: mensagens_gigante.append({"role": "user", "content": str(item[0])})
                    if item[1]: mensagens_gigante.append({"role": "assistant", "content": str(item[1])})
            mensagens_gigante.append({"role": "user", "content": texto_final})
            
            resposta = chamar_gigante_openrouter(mensagens_gigante)
            if resposta: 
                salvar_na_memoria(f"USER: {texto_usuario}\nIA: {resposta}", "Chat (Gigante)")
                return resposta

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
        salvar_na_memoria(f"USER: {texto_usuario}\nIA: {resposta}", "Chat (GPT-OSS)")
        return resposta
        
    except Exception as e: return f"⚠️ Falha de Conexão. Detalhe técnico: {str(e)}"

def exportar_conversa(historico):
    if not historico: return None
    pasta = f"{DIR_CASOS}/Chat_{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(pasta, exist_ok=True)
    cam_word = f"{pasta}/Protocolo_O_Codigo_de_Ouro.docx"
    doc = docx.Document()
    doc.add_heading('Protocolo - O Código de Ouro', 0)
    for item in historico:
        if isinstance(item, dict):
            doc.add_heading("Usuário:" if item.get("role") == "user" else "O Código de Ouro:", level=2)
            doc.add_paragraph(str(item.get("content")))
    doc.save(cam_word)
    return cam_word

# ==========================================
# 6. AGENTE MESTRE, DOSSIÊ E ESTÚDIO
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
        
        regra_tribunal = "\n\n[SISTEMA DE DUPLA CHECAGEM ATIVADO]: Você deve agir como um Auditor Implacável. Revise os dados cruzados. Se houver qualquer divergência, contradição ou anomalia nas informações extraídas, crie obrigatoriamente uma seção destacada no texto final chamada '⚠️ ALERTA DE DIVERGÊNCIA' e aponte o erro exato para segurança do cliente." if usar_tribunal else ""
        regra_imagem = "\nNo final, escreva: [IMAGEM: descreva em INGLÊS uma cena fotorrealista para este conteúdo]" if usar_img else ""
        
        prompt = f"Analise os dados fornecidos como a inteligência primária do sistema O Código de Ouro.\nDADOS: {contexto}\nAÇÃO: {instrucao}{regra_tribunal}{regra_imagem}"
        
        resposta = chamar_gigante_openrouter([{"role": "user", "content": prompt}]) if chave_openrouter else None
        if not resposta: resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ_RAPIDO, max_tokens=4000).choices[0].message.content
        
        cam_img = None
        if usar_img:
            progresso(0.8, desc="Renderizando Arte...")
            match = re.search(r'\[IMAGEM:\s*(.*?)\]', resposta, re.IGNORECASE)
            resposta_limpa = re.sub(r'\[IMAGEM:\s*(.*?)\]', '', resposta, flags=re.IGNORECASE).strip()
            cam_img = f"{pasta}/Capa_Projeto.jpg"
            cliente_hf.text_to_image(match.group(1).strip() if match else "Corporate cover, gold", model="black-forest-labs/FLUX.1-schnell").save(cam_img)
        else: resposta_limpa = resposta

        progresso(0.9, desc="Exportando...")
        cam_word = f"{pasta}/Relatorio_Codigo_de_Ouro.docx"
        doc = docx.Document()
        doc.add_heading('Relatório - O Código de Ouro', 0)
        if cam_img: doc.add_picture(cam_img, width=Inches(6.0))
        doc.add_paragraph(resposta_limpa)
        doc.save(cam_word)
        salvar_na_memoria(resposta_limpa, "Auditoria")
        
        cam_audio = f"{pasta}/Audio_Codigo_Ouro.mp3"
        if usar_aud:
            with open(f"{pasta}/temp.txt", "w", encoding="utf-8") as f: f.write(resposta_limpa[:3000].replace('*', ''))
            os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{pasta}/temp.txt" --write-media "{cam_audio}"')
        return "✅ Auditoria Concluída", cam_word, cam_audio if usar_aud else None, resposta_limpa, f"📊 STATUS: {palavras} palavras validadas."
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
# 7. TIPOGRAFIA DE ELITE E DESIGN SYSTEM
# ==========================================
tema_ultra = gr.themes.Base(
    font=[gr.themes.GoogleFont("Outfit"), gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#070707", body_background_fill_dark="#070707", 
    body_text_color="#F5F5F7", body_text_color_dark="#F5F5F7", 
    background_fill_primary="#0D0D0D", background_fill_primary_dark="#0D0D0D", 
    background_fill_secondary="#111111", background_fill_secondary_dark="#111111",
    border_color_primary="#2A2A2A", border_color_primary_dark="#2A2A2A",
    block_background_fill="#0D0D0D", block_background_fill_dark="#0D0D0D",
    block_label_text_color="#C5A059", block_label_text_color_dark="#C5A059", 
    block_title_text_color="#D4AF37", block_title_text_color_dark="#D4AF37", 
    input_background_fill="#121212", input_background_fill_dark="#121212", 
    input_border_color="#333333", input_border_color_dark="#333333",
    button_primary_background_fill="linear-gradient(145deg, #D4AF37, #AA7C11)", button_primary_background_fill_dark="linear-gradient(145deg, #D4AF37, #AA7C11)",
    button_primary_text_color="#000000", button_secondary_background_fill="#181818", button_secondary_text_color="#C5A059"
)

css_ultra = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;800&display=swap');

body, .gradio-container { background-color: #070707 !important; color: #F5F5F7 !important; font-family: 'Inter', sans-serif !important; }
h1, h2, h3, h4, h5, h6, .tab-nav button { font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; }

footer { display: none !important; }

p, label, .markdown-text, .chatbot { color: #E0E0E0 !important; font-size: 1.02rem !important; line-height: 1.6 !important; }
h3 { color: #C5A059 !important; letter-spacing: 1px; }

code, pre { background-color: #181818 !important; color: #D4AF37 !important; border: 1px solid #2A2A2A !important; border-radius: 6px !important; font-family: monospace !important; padding: 2px 6px !important; }

input:-webkit-autofill { -webkit-box-shadow: 0 0 0 30px #121212 inset !important; -webkit-text-fill-color: #F5F5F7 !important; }
textarea, input, select, .wrap-inner, .dropdown-menu, .wrap { background-color: #121212 !important; color: #F5F5F7 !important; border: 1px solid #333333 !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important; }
textarea::placeholder, input::placeholder { color: #555555 !important; }

button { text-transform: uppercase; font-weight: 700 !important; letter-spacing: 1.5px !important; transition: 0.3s all ease !important; border-radius: 8px !important; }
button:hover { transform: translateY(-2px); }
button.primary { color: #000000 !important; border: none !important; box-shadow: 0 4px 15px rgba(212, 175, 55, 0.25) !important; text-shadow: none !important; font-family: 'Outfit', sans-serif !important; }
button.secondary { border: 1px solid #3A3A3A !important; background: #181818 !important; color: #D4AF37 !important; font-family: 'Outfit', sans-serif !important; }

.tabs { background: transparent !important; border: none !important; }
.tab-nav { background: transparent !important; border-bottom: 1px solid #222 !important; padding: 0 20px !important; gap: 15px; justify-content: flex-start !important; }
.tab-nav button { color: #666666 !important; padding: 15px 25px !important; border-radius: 0 !important; border: none !important; background: transparent !important; border-bottom: 2px solid transparent !important; }
.tab-nav button.selected { color: #D4AF37 !important; border-bottom: 2px solid #D4AF37 !important; background: transparent !important; text-shadow: 0 0 10px rgba(212,175,55,0.3) !important; }

.box-painel { background: #0D0D0D !important; border-radius: 12px !important; padding: 30px !important; border: 1px solid #1F1F1F !important; margin-bottom: 25px; box-shadow: 0 15px 35px rgba(0,0,0,0.4); }

.chat-container { border-radius: 12px !important; background: #0A0A0A !important; border: 1px solid #222 !important; }
.message.bot { background: #111111 !important; border-left: 2px solid #D4AF37 !important; color: #F5F5F7 !important; border-radius: 0 12px 12px 0 !important; }
.message.user { background: #181818 !important; border: 1px solid #2A2A2A !important; color: #C5A059 !important; border-radius: 12px 12px 0 12px !important; }

.sidebar { background: #070707 !important; border-right: 1px solid #1F1F1F !important; padding: 25px !important; }
"""

with gr.Blocks(title="O Código de Ouro", theme=tema_ultra, css=css_ultra, fill_width=True, js="function() { document.body.classList.add('dark'); }") as interface:
    
    with gr.Sidebar(open=True):
        gr.HTML(renderizar_logo())
        status_cerebro = "🔵 **Módulo: 70 Bilhões de Parâmetros**" if os.environ.get("OPENROUTER_API_KEY") else "🟢 **Módulo: GPT-OSS / Qwen**"
        gr.HTML("<h3 style='text-align: center; color: #555; text-transform: uppercase; font-size: 11px; letter-spacing: 4px; margin-bottom: 20px;'>Inteligência Suprema</h3>")
        
        with gr.Accordion("⚙️ Engine Operacional", open=False):
            gr.Markdown(f"{status_cerebro}")
            persona_box = gr.Dropdown(choices=["Assessor Gold (Padrão)", "Estrategista de Vendas", "Auditor de Negócios", "Diretor Criativo"], value="Assessor Gold (Padrão)", show_label=False)
            net_box = gr.Checkbox(label="🌐 Conexão Web (Busca)", value=False)
            btn_limpa = gr.Button("🧹 Purgar Memória", variant="secondary")
            msg_sys = gr.Textbox(show_label=False, interactive=False)
            btn_limpa.click(fn=limpar_banco_de_dados, outputs=msg_sys)
        
        with gr.Accordion("📡 Webhooks de Automação", open=False):
            gr.Markdown("*Transmissão via Zapier/Make*")
            url_webhook = gr.Textbox(placeholder="URL do Webhook...", show_label=False)
            texto_export = gr.Textbox(placeholder="Conteúdo a enviar...", lines=2, show_label=False)
            btn_web = gr.Button("Disparar Transmissão", variant="primary")
            msg_web = gr.Textbox(show_label=False, interactive=False)
            btn_web.click(fn=disparar_webhook, inputs=[url_webhook, texto_export], outputs=msg_web)

        gr.HTML("<hr style='border: none; border-bottom: 1px solid #1F1F1F; margin: 30px 0;'>")
        btn_back = gr.Button("📦 Extrair Cofre Geral", variant="primary")
        msg_b = gr.Textbox(show_label=False)
        arq_b = gr.File(label="Arquivo", visible=False)
        btn_back.click(fn=gerar_backup, outputs=[arq_b, msg_b]).then(lambda: gr.update(visible=True), None, arq_b)

    with gr.Tabs():
        
        with gr.TabItem("🧠 O CÓDIGO DE OURO"):
            chat = gr.ChatInterface(
                fn=responder_chat_multimodal, multimodal=True, additional_inputs=[persona_box, net_box],
                chatbot=gr.Chatbot(height="70vh", show_label=False, placeholder="SISTEMA ATIVO. INSIRA SUA DIRETRIZ."),
                textbox=gr.MultimodalTextbox(placeholder="Comandos, planilhas, PDFs ou imagens...", container=False, scale=7, show_label=False)
            )

        with gr.TabItem("🤖 AGENTE MESTRE"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    gr.Markdown("### EXECUÇÃO AUTÔNOMA\nDelegue um objetivo complexo. O agente efetuará pesquisas de mercado, redigirá a estratégia e renderizará ativos visuais.")
                    txt_missao = gr.Textbox(label="Diretriz da Missão", lines=5, placeholder="Ex: Desenvolva uma campanha para um relógio de luxo...")
                    btn_agente = gr.Button("INICIAR PROTOCOLO", variant="primary", size="lg")
                with gr.Column(scale=6):
                    out_estrat = gr.Textbox(label="Estratégia Sintetizada", lines=16, interactive=False)
                    out_arte = gr.Image(label="Ativo Visual Comercial", type="filepath")
            btn_agente.click(fn=executar_agente_mestre, inputs=[txt_missao], outputs=[out_estrat, out_arte])

        with gr.TabItem("📑 AUDITORIA IA"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    arq_up = gr.File(label="Cofre de Documentos (PDF/Imagens/XLSX)", file_count="multiple")
                    txt_ordem = gr.Textbox(label="Diretriz de Auditoria", lines=3, placeholder="Determine o alvo da análise documental...")
                    with gr.Row():
                        c_img = gr.Checkbox(label="🖼️ Capa Visual", value=False)
                        c_aud = gr.Checkbox(label="🔊 Síntese Vocal", value=False)
                        c_trib = gr.Checkbox(label="⚖️ Checagem Antifraude", value=False)
                    btn_exe = gr.Button("INICIAR VARREDURA NEURAL", variant="primary")
                with gr.Column(scale=6):
                    out_tela = gr.Textbox(label="Dossiê Preliminar", lines=20, interactive=False)
                    with gr.Row():
                        out_word = gr.File(label="Dossiê Oficial (Word)")
                        out_aud = gr.Audio(label="Ouvir Laudo Executivo")
                    out_tel = gr.Textbox(show_label=False, lines=1, interactive=False)
            btn_exe.click(fn=gerar_dossie, inputs=[arq_up, txt_ordem, c_img, c_aud, c_trib], outputs=[msg_sys, out_word, out_aud, out_tela, out_tel])

        with gr.TabItem("🎬 ESTÚDIO GOLD"):
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37;'>🖼️ FOTOGRAFIA COMERCIAL</h3>")
                    img_sujeito = gr.Textbox(label="Foco da Composição", placeholder="Ex: Um frasco de perfume dourado...")
                    img_fundo = gr.Textbox(label="Atmosfera de Fundo")
                    img_estilo = gr.Dropdown(choices=["Fotorrealista 8k", "Cinematic Dark", "Cyberpunk", "Minimalista de Luxo"], label="Direção de Arte", value="Fotorrealista 8k")
                    btn_gerar_img = gr.Button("RENDERIZAR COMPOSIÇÃO", variant="primary")
                    out_img_est = gr.Image(label="Ativo Final", type="filepath")
                    btn_gerar_img.click(fn=gerar_imagem_estudio, inputs=[img_sujeito, img_fundo, img_estilo], outputs=[out_img_est])
                
                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37;'>🎙️ SÍNTESE VOCAL NEURAL</h3>")
                    txt_aud = gr.Textbox(show_label=False, placeholder="Insira o roteiro da campanha aqui...", lines=5)
                    btn_gerar_aud = gr.Button("SINTETIZAR LOCUÇÃO", variant="primary")
                    out_aud_estudio = gr.Audio(label="Arquivo Master (MP3)")
                    btn_gerar_aud.click(fn=falar_laudo_estudio, inputs=[txt_aud], outputs=[out_aud_estudio])

# ==========================================
# GESTÃO DE USUÁRIOS E TELA DE LOGIN
# ==========================================
lista_de_usuarios = []
for i in ["", "_1", "_2", "_3", "_4", "_5"]:
    u = os.environ.get(f"LOGIN_USUARIO{i}") or os.environ.get(f"USUARIO{i}")
    s = os.environ.get(f"LOGIN_SENHA{i}") or os.environ.get(f"SENHA{i}")
    if u and s:
        lista_de_usuarios.append((u, s))

html_tela_login = renderizar_logo_login() if lista_de_usuarios else None

interface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 10000)), 
    auth=lista_de_usuarios if lista_de_usuarios else None,
    auth_message=html_tela_login
)
