import gradio as gr
from gradio_client import Client
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

# Contingência Global para Embeddings
try:
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=chave_hf, model_name="sentence-transformers/all-MiniLM-L6-v2")
except: embeddings = None

# ==========================================
# 1.5 RADAR DE CÓRTEX (AUTO-DESCOBERTA)
# ==========================================
MODELOS_TEXTO_ATIVOS = []
MODELOS_VISAO_ATIVOS = []

if chave_groq:
    try:
        resposta_modelos = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {chave_groq}"}, timeout=10).json()
        if "data" in resposta_modelos:
            for m in resposta_modelos["data"]:
                m_id = m["id"]
                if "whisper" in m_id.lower() or "tool-use" in m_id.lower(): continue
                if "vision" in m_id.lower(): MODELOS_VISAO_ATIVOS.append(m_id)
                else: MODELOS_TEXTO_ATIVOS.append(m_id)
    except: pass

if not MODELOS_TEXTO_ATIVOS: MODELOS_TEXTO_ATIVOS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
if not MODELOS_VISAO_ATIVOS: MODELOS_VISAO_ATIVOS = ["llama-3.2-90b-vision-preview"]

# ==========================================
# 2. DIRETÓRIOS E BANCO DE DADOS (PERSISTÊNCIA)
# ==========================================
DIRETORIO = "./Central_IA_Master"
DIR_CHROMA = f"{DIRETORIO}/Banco_de_Dados_Vetorial"
DIR_CASOS = f"{DIRETORIO}/Projetos_Salvos"
DIR_MIDIA = f"{DIRETORIO}/Midia_Criada"
DB_PATH = f"{DIRETORIO}/codigo_de_ouro.db"

for d in [DIRETORIO, DIR_CASOS, DIR_MIDIA]:
    try: os.makedirs(d, exist_ok=True)
    except: pass

def iniciar_banco():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS memoria_sessoes (id INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT, conteudo TEXT, tipo TEXT)''')
        conn.commit()
        conn.close()
    except: pass

iniciar_banco()

def salvar_na_memoria(conteudo, tipo="Geral"):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO memoria_sessoes (data, conteudo, tipo) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(conteudo), tipo))
        conn.commit()
        conn.close()
    except: pass

def ler_memoria():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT id as 'ID', data as 'DATA', tipo as 'MÓDULO', conteudo as 'REGISTRO DE OPERAÇÃO' FROM memoria_sessoes ORDER BY id DESC LIMIT 20", conn)
        conn.close()
        return df
    except: return pd.DataFrame(columns=["SISTEMA", "STATUS", "MENSAGEM"], data=[["Caixa Preta", "Operante", "Nenhum registro recuperado."]])

# ==========================================
# 3. RENDERIZAÇÃO DA LOGO E DESIGN
# ==========================================
def buscar_caminho_logo():
    try:
        for root, dirs, files in os.walk("."):
            for f in files:
                if "chamariz" in f.lower() and "fundo" in f.lower(): return os.path.join(root, f)
    except: pass
    return None

def renderizar_logo():
    caminho = buscar_caminho_logo()
    if caminho:
        try:
            extensao = "png" if caminho.lower().endswith(".png") else "jpeg"
            with open(caminho, "rb") as f: encoded = base64.b64encode(f.read()).decode('utf-8')
            return f'''<div style="display: flex; justify-content: center; align-items: center; padding: 15px 0 30px 0;"><img src="data:image/{extensao};base64,{encoded}" style="max-width: 75%; filter: drop-shadow(0px 8px 20px rgba(212, 175, 55, 0.4));"></div>'''
        except: pass
    return '''<div style="text-align: center; margin-bottom: 30px; padding: 25px 10px; border-radius: 12px; background: linear-gradient(145deg, #BF953F, #B38728); box-shadow: 0 10px 25px rgba(212,175,55,0.2);"><h1 style="color: #000; font-family: 'Outfit', sans-serif; font-weight: 800; letter-spacing: 4px; margin: 0; font-size: 20px;">O CÓDIGO DE OURO</h1></div>'''

def renderizar_logo_login():
    caminho = buscar_caminho_logo()
    if caminho:
        try:
            extensao = "png" if caminho.lower().endswith(".png") else "jpeg"
            with open(caminho, "rb") as f: encoded = base64.b64encode(f.read()).decode('utf-8')
            return f'''<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100%; padding: 10px 0 25px 0;"><img src="data:image/{extensao};base64,{encoded}" style="width: 380px; max-width: 90%; margin: 0 auto 15px auto; display: block; filter: drop-shadow(0px 8px 20px rgba(212, 175, 55, 0.6));"><span style="color: #D4AF37; font-family: 'Outfit', sans-serif; letter-spacing: 5px; font-size: 13px; font-weight: 700; text-transform: uppercase;">Acesso Restrito</span></div>'''
        except: pass
    return '''<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100%; padding: 10px 0 25px 0;"><h1 style="color: #D4AF37; font-family: 'Outfit', sans-serif; letter-spacing: 4px; font-size: 26px; margin-bottom: 10px; text-align: center;">O CÓDIGO DE OURO</h1><span style="color: #888; font-family: 'Inter', sans-serif; letter-spacing: 3px; font-size: 12px; font-weight: 600; text-align: center;">ACESSO RESTRITO</span></div>'''

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file: return base64.b64encode(image_file.read()).decode('utf-8')
    except: return ""

# ==========================================
# 4. MOTOR HÍBRIDO ESTRATÉGICO
# ==========================================
def motor_neural_groq(mensagens, visao=False, max_tokens=3000):
    modelos_alvo = MODELOS_VISAO_ATIVOS if visao else MODELOS_TEXTO_ATIVOS
    erros_acumulados = []
    
    for mod in modelos_alvo:
        try:
            r = cliente_groq.chat.completions.create(model=mod, messages=mensagens, max_tokens=max_tokens, temperature=0.3)
            if r and r.choices: return r.choices[0].message.content
        except Exception as e: 
            erros_acumulados.append(f"[{mod}: {str(e)[:50]}]")
            continue
            
    log_erros = " | ".join(erros_acumulados)
    salvar_na_memoria(f"Falha Geral APIs: {log_erros}", "Sistema Erro")
    return f"⚠️ [SISTEMA DE CONTINGÊNCIA ATIVADO]: O Córtex Neural encontrou bloqueios nos nós externos. Diagnóstico: {log_erros}"

def chamar_gigante_openrouter(mensagens):
    if not chave_openrouter: return None
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {chave_openrouter}", "Content-Type": "application/json", "HTTP-Referer": "https://ocodigodeouro.com"}, json={"model": "meta-llama/llama-3.3-70b-instruct:free", "messages": mensagens, "max_tokens": 4000}, timeout=25)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"]
    except: return None
    return None

# ==========================================
# 5. EXTRAÇÃO NEURAL SEGURA
# ==========================================
def extrair_texto(arquivo):
    caminho = arquivo.name if hasattr(arquivo, 'name') else arquivo
    nome = caminho.lower()
    texto = ""
    try:
        if nome.endswith('.pdf'):
            with pdfplumber.open(caminho) as pdf:
                for idx, p in enumerate(pdf.pages):
                    try:
                        txt_digital = p.extract_text()
                        if txt_digital and len(txt_digital.strip()) > 30: texto += txt_digital + "\n"
                        else:
                            if idx < 5: 
                                buffer = io.BytesIO()
                                p.to_image(resolution=150).original.save(buffer, format="JPEG")
                                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                if img_b64:
                                    msg = [{"role": "user", "content": [{"type": "text", "text": "Extract text."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}]
                                    ext = motor_neural_groq(msg, visao=True, max_tokens=1000)
                                    if not ext.startswith("⚠️"): texto += ext + "\n"
                                    time.sleep(1.5) 
                    except: continue 
        elif nome.endswith(('.xlsx', '.csv')): texto = (pd.read_excel(caminho) if nome.endswith('.xlsx') else pd.read_csv(caminho)).to_string() 
        elif nome.endswith('.docx'): texto += "\n".join([p.text for p in docx.Document(caminho).paragraphs])
        elif nome.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            try:
                img_b64 = encode_image(caminho)
                if img_b64:
                    msg = [{"role": "user", "content": [{"type": "text", "text": "Extract all data."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}]
                    ext = motor_neural_groq(msg, visao=True, max_tokens=1500)
                    if not ext.startswith("⚠️"): texto += f"[IMAGEM: {os.path.basename(caminho)}]\n" + ext + "\n"
            except: pass
    except: pass
    return texto

def limpar_banco_de_dados():
    try:
        if os.path.exists(DIR_CHROMA): shutil.rmtree(DIR_CHROMA)
        return "🧹 Memória Purgada com Sucesso."
    except Exception as e: return f"Erro não crítico: {e}"

def aprimorar_prompt(sujeito, fundo, estilo):
    sys_msg = "You are an AI prompt engineer. Translate the user's Portuguese concept into a highly detailed English prompt. CRITICAL: Output ONLY the raw English prompt."
    user_msg = f"Translate and enhance:\nFocus: {sujeito}\nBackground: {fundo}\nStyle: {estilo}\nAdd tags: 8k resolution, highly detailed, photorealistic."
    try: 
        res = motor_neural_groq([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}], max_tokens=150)
        if res and not res.startswith("⚠️"): return res.replace('"', '').replace('Here is the prompt:', '').strip()
        return f"{sujeito}, {fundo}, {estilo}" 
    except: return f"{sujeito}, {fundo}, {estilo}"

def gerar_imagem_estudio(sujeito, fundo, estilo):
    if not sujeito: return None
    c = f"{DIR_MIDIA}/Img_{datetime.now().strftime('%H%M%S')}.jpg"
    try:
        prompt_en = aprimorar_prompt(sujeito, fundo, estilo)
        cliente_hf.text_to_image(prompt_en, model="black-forest-labs/FLUX.1-schnell").save(c)
        return c
    except Exception as e: 
        salvar_na_memoria(f"Falha ao renderizar imagem: {e}", "Erro Mídia")
        return None

def gerar_video_ia(imagem_base, sujeito, fundo, movimento):
    try:
        if imagem_base:
            return Client("multimodalart/stable-video-diffusion").predict(imagem_base, api_name="/video"), "✅ Cena animada com êxito."
        if not sujeito: return None, "⚠️ O Sujeito é obrigatório."
        
        prompt_en = aprimorar_prompt(sujeito, fundo, movimento)
        try:
            return Client("multimodalart/zeroscope-v2").predict(prompt_en, api_name="/infer"), "✅ Masterização Finalizada."
        except:
            return Client("hysts/ModelScope-Text-To-Video-Synthesis").predict(prompt_en, api_name="/infer"), "✅ Masterização Finalizada (Rota Alternativa)."
            
    except Exception as e: 
        salvar_na_memoria(f"Falha Motor Vídeo: {e}", "Erro Mídia")
        return None, f"⚠️ Acesso restrito global: Os servidores públicos de vídeo da Hugging Face atingiram o limite. Detalhe: {str(e)[:40]}"

def falar_laudo_estudio(texto):
    if not texto: return None
    c = f"{DIR_MIDIA}/Voz_{datetime.now().strftime('%H%M%S')}.mp3"
    try:
        with open("t.txt", "w", encoding="utf-8") as f: f.write(texto[:3000].replace('*', ''))
        os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "t.txt" --write-media "{c}"')
        return c
    except: return None

def gerar_backup():
    cam = "./Cofre_Master_Backup.zip"
    try:
        with zipfile.ZipFile(cam, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(DIRETORIO):
                if "Banco_de_Dados_Vetorial" not in root:
                    for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), DIRETORIO))
        return cam, "📦 Pacote de Extração Gerado."
    except Exception as e: return None, f"⚠️ Erro ao empacotar: {e}"

# ==========================================
# 6. MÓDULOS DE CHAT OMNI-COMMAND
# ==========================================
def disparar_webhook(url, texto_contexto):
    if not url: return "⚠️ Requisito pendente: Informe a URL."
    if not texto_contexto: return "⚠️ Pacote vazio."
    try:
        r = requests.post(url, json={"sistema": "O Código de Ouro", "dados": texto_contexto, "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, timeout=15)
        return "✅ Protocolo Executado!" if r.status_code == 200 else f"✅ Sinal disparado. Retorno: {r.status_code}"
    except Exception as e: return f"⚠️ Falha Crítica: {e}"

def responder_chat_multimodal(mensagem, historico, persona, usar_internet):
    try:
        texto_usuario = mensagem.get("text", "") if isinstance(mensagem, dict) else str(mensagem)
        arquivos = mensagem.get("files", []) if isinstance(mensagem, dict) else []
        contexto_extra, imagens = "", []
        
        for arq in arquivos:
            cam = arq["path"] if isinstance(arq, dict) and "path" in arq else (arq if isinstance(arq, str) else getattr(arq, 'name', ''))
            if not cam: continue
            if cam.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): imagens.append(cam)
            else: contexto_extra += f"\n[ACERVO ANEXADO]:\n{extrair_texto(cam)}\n"
                
        if usar_internet and texto_usuario:
            try:
                buscas = DDGS(timeout=10).text(texto_usuario, max_results=4)
                if buscas: contexto_extra += "\n\n[MAPEAMENTO WEB]:\n" + "\n".join([f"FONTE ({r['title']}): {r['body']}" for r in buscas])
            except: contexto_extra += f"\n\n[AVISO]: Antena Web bloqueada temporariamente."

        sys_prompt = f"Você é a Inteligência de Comando do sistema 'O Código de Ouro', atuando como {persona}. "
        sys_prompt += "DIRETRIZ SUPREMA: Se houver dados marcados como [MAPEAMENTO WEB], assuma-os como a verdade do presente. Não mencione limitações de data.\n\n"
        sys_prompt += "HABILIDADES OMNI (GERAÇÃO DIRETA NO CHAT):\n"
        sys_prompt += "- Se o usuário pedir para gerar, criar ou desenhar uma IMAGEM ou FOTO, adicione no final da sua resposta: [IMG: descrição da cena em inglês com detalhes fotorrealistas]\n"
        sys_prompt += "- Se o usuário pedir para gerar ou criar um VÍDEO, adicione no final: [VID: descrição da cena em inglês para animação]\n"
        sys_prompt += "- Se o usuário pedir para gerar um DOCUMENTO ou LAUDO, adicione no final: [DOC: insira aqui o texto e conteúdo completo que deve ir para dentro do documento]"
        
        texto_final = texto_usuario + contexto_extra
        if imagens and not texto_final.strip(): texto_final = "Realize uma varredura pericial absoluta nesta imagem."
        elif not imagens and not texto_final.strip(): return "⚠️ Terminal ocioso. Insira uma diretriz de comando."

        mensagens_ia = [{"role": "system", "content": sys_prompt}]
        for item in historico:
            if isinstance(item, dict) and item.get("content"): mensagens_ia.append({"role": item.get("role"), "content": str(item.get("content"))})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                if item[0]: mensagens_ia.append({"role": "user", "content": str(item[0])})
                if item[1]: mensagens_ia.append({"role": "assistant", "content": str(item[1])})

        if imagens:
            conteudo_msg = [{"type": "text", "text": texto_final}]
            for img in imagens: 
                try:
                    img_enc = encode_image(img)
                    if img_enc: conteudo_msg.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_enc}"}})
                except: pass
            mensagens_ia.append({"role": "user", "content": conteudo_msg})
            resposta = motor_neural_groq(mensagens_ia, visao=True)
        else:
            mensagens_ia.append({"role": "user", "content": texto_final})
            resposta = chamar_gigante_openrouter(mensagens_ia) if chave_openrouter else None
            if not resposta: resposta = motor_neural_groq(mensagens_ia, visao=False)

        if not resposta or resposta.startswith("⚠️"):
            salvar_na_memoria(f"Falha Chat: {resposta}", "Terminal Erro")
            return resposta if resposta else "⚠️ Falha crítica ao gerar resposta neural."

        if "[IMG:" in resposta:
            for match in re.finditer(r'\[IMG:\s*(.*?)\]', resposta, re.IGNORECASE):
                try:
                    prompt_img = match.group(1)
                    cam = gerar_imagem_estudio(prompt_img, "", "Fotorrealista 8k")
                    if cam:
                        tag_md = f"\n\n🖼️ **Ativo Visual Renderizado:**\n![Imagem](/file={os.path.abspath(cam)})"
                        resposta = resposta.replace(match.group(0), tag_md)
                    else: resposta = resposta.replace(match.group(0), "\n\n⚠️ [Contingência]: Renderizador de Imagens sobrecarregado.")
                except: resposta = resposta.replace(match.group(0), "")
                    
        if "[VID:" in resposta:
            for match in re.finditer(r'\[VID:\s*(.*?)\]', resposta, re.IGNORECASE):
                try:
                    prompt_vid = match.group(1)
                    cam, msg = gerar_video_ia(None, prompt_vid, "", "Cinematic Tracking")
                    if cam:
                        tag_html = f"\n\n🎥 **Vídeo Masterizado:**\n<video controls width='100%'><source src='/file={os.path.abspath(cam)}' type='video/mp4'></video>"
                        resposta = resposta.replace(match.group(0), tag_html)
                    else:
                        resposta = resposta.replace(match.group(0), f"\n\n⚠️ Erro de Renderização de Vídeo: {msg}")
                except Exception as e: resposta = resposta.replace(match.group(0), f"\n\n⚠️ Motor de Vídeo Offline.")
                    
        if "[DOC:" in resposta:
            try:
                match_doc = re.search(r'\[DOC:\s*(.*?)\]', resposta, re.IGNORECASE | re.DOTALL)
                if match_doc:
                    conteudo_doc = match_doc.group(1)
                    pasta_doc = f"{DIR_CASOS}/Dossie_Omni_{datetime.now().strftime('%d%m_%H%M%S')}"
                    os.makedirs(pasta_doc, exist_ok=True)
                    cam_word = f"{pasta_doc}/Documento_Oficial_Gerado.docx"
                    doc_arquivo = docx.Document()
                    doc_arquivo.add_heading('Dossiê - O Código de Ouro', 0)
                    doc_arquivo.add_paragraph(conteudo_doc.strip())
                    doc_arquivo.save(cam_word)
                    tag_doc = f"\n\n📄 **Documento Formatado Pronto:**\n[📥 Clique aqui para baixar o Relatório Word (.docx)](/file={os.path.abspath(cam_word)})"
                    resposta = resposta.replace(match_doc.group(0), tag_doc)
            except: pass

        salvar_na_memoria(f"U: {texto_usuario[:50]}... IA: Processamento Omni Ativo.", "Terminal Master")
        return resposta
    except Exception as e: return f"⚠️ Interrupção no Córtex do Sistema: {str(e)}"

# ==========================================
# 7. AGENTE MESTRE E ESTÚDIO ISOLADOS
# ==========================================
def executar_agente_mestre(objetivo, progresso=gr.Progress()):
    if not objetivo: return "⚠️ Parâmetros insuficientes. Defina o alvo estratégico.", None
    try:
        progresso(0.3, desc="🔍 Engajando Protocolos...")
        try: ctx = "\n".join([r['body'] for r in DDGS(timeout=10).text(objetivo, max_results=4)])
        except: ctx = ""
        
        progresso(0.6, desc="🧠 Sintetizando Matriz...")
        prompt = f"Missão Alfa: '{objetivo}'. Mercado: {ctx}. Crie plano tático. Adicione: [IMAGEM: descrição em inglês fotorrealista]."
        estrategia = chamar_gigante_openrouter([{"role": "user", "content": prompt}]) if chave_openrouter else None
        if not estrategia: estrategia = motor_neural_groq([{"role": "user", "content": prompt}])
        if estrategia.startswith("⚠️"): return estrategia, None
        
        progresso(0.9, desc="🎨 Renderizando Arte...")
        cam_img = None
        try:
            match = re.search(r'\[IMAGEM:\s*(.*?)\]', estrategia, re.IGNORECASE)
            resposta_limpa = re.sub(r'\[IMAGEM:\s*(.*?)\]', '', estrategia, flags=re.IGNORECASE).strip()
            cam_img = f"{DIR_MIDIA}/Agente_{datetime.now().strftime('%H%M%S')}.jpg"
            cliente_hf.text_to_image(match.group(1).strip() if match else f"High-end corporate asset for {objetivo}, 8k", model="black-forest-labs/FLUX.1-schnell").save(cam_img)
        except: resposta_limpa = estrategia 
        
        salvar_na_memoria(objetivo[:50], "Agente Autônomo")
        progresso(1.0, desc="✅ Missão Cumprida.")
        return resposta_limpa, cam_img
    except Exception as e: return f"⚠️ Falha não fatal no Agente: {e}", None

def gerar_dossie(arquivos, instrucao, usar_img, usar_aud, usar_tribunal, progresso=gr.Progress()):
    if not instrucao: return "⚠️ É necessário definir uma diretriz de análise.", None, None, "", ""
    palavras = 0
    try:
        progresso(0.2, desc="Desfragmentando Acervo...")
        pasta = f"{DIR_CASOS}/Dossie_{datetime.now().strftime('%d%m_%H%M')}"
        os.makedirs(pasta, exist_ok=True)
        if not embeddings: return "⚠️ Erro de Banco de Dados: Motor de Incorporação Offline.", None, None, "", ""
        
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for arq in arquivos:
                txt = extrair_texto(arq) 
                palavras += len(txt.split())
                if txt.strip(): banco.add_texts([f"[ORIGEM: {os.path.basename(arq.name)}]\n{c}" for c in fatiador.split_text(txt)])
            
        progresso(0.5, desc="Sintetizando Laudo...")
        try: contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        except: contexto = ""
        
        regra_tribunal = "\n[ANTIFRAUDE]: Aponte contradições matemáticas em '⚠️ ALERTA DE DIVERGÊNCIA'." if usar_tribunal else ""
        regra_img = "\n[IMAGEM: descrição em inglês fotorrealista para capa]" if usar_img else ""
        
        prompt = f"Protocolo Auditor.\n\n[DADOS]: {contexto}\n\n[AÇÃO]: {instrucao}{regra_tribunal}{regra_img}"
        resposta = chamar_gigante_openrouter([{"role": "user", "content": prompt}]) if chave_openrouter else None
        if not resposta: resposta = motor_neural_groq([{"role": "user", "content": prompt}])
        if resposta.startswith("⚠️"): return resposta, None, None, "", ""
        
        cam_img = None
        resposta_limpa = resposta
        if usar_img:
            progresso(0.8, desc="Gerando Capa...")
            try:
                match = re.search(r'\[IMAGEM:\s*(.*?)\]', resposta, re.IGNORECASE)
                resposta_limpa = re.sub(r'\[IMAGEM:\s*(.*?)\]', '', resposta, flags=re.IGNORECASE).strip()
                cam_img = f"{pasta}/Capa.jpg"
                cliente_hf.text_to_image(match.group(1).strip() if match else "Dark luxury corporate dossier cover", model="black-forest-labs/FLUX.1-schnell").save(cam_img)
            except: resposta_limpa = resposta
        
        progresso(0.9, desc="Compilando Docx...")
        cam_word = f"{pasta}/Laudo.docx"
        try:
            doc = docx.Document()
            doc.add_heading('Laudo de Auditoria - O Código de Ouro', 0)
            if cam_img and os.path.exists(cam_img): doc.add_picture(cam_img, width=Inches(6.0))
            doc.add_paragraph(resposta_limpa)
            doc.save(cam_word)
        except: cam_word = None
        
        salvar_na_memoria("Auditoria: " + instrucao[:30], "Tribunal")
        
        cam_audio = f"{pasta}/Audio.mp3" if usar_aud else None
        if usar_aud:
            try:
                with open(f"{pasta}/t.txt", "w", encoding="utf-8") as f: f.write(resposta_limpa[:3000].replace('*', ''))
                os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{pasta}/t.txt" --write-media "{cam_audio}"')
            except: cam_audio = None
            
        return "✅ Auditoria Completa", cam_word, cam_audio, resposta_limpa, f"📊 MÉTRICA: {palavras} palavras processadas."
    except Exception as e: return f"⚠️ Falha de Segurança Operacional: {e}", None, None, "", ""

def falar_laudo_estudio(texto):
    if not texto: return None
    c = f"{DIR_MIDIA}/Voz_{datetime.now().strftime('%H%M%S')}.mp3"
    try:
        with open("t.txt", "w", encoding="utf-8") as f: f.write(texto[:3000].replace('*', ''))
        os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "t.txt" --write-media "{c}"')
        return c
    except: return None

# ==========================================
# 8. DESIGN SYSTEM ALTO LUXO (COM SCROLL FIXO)
# ==========================================
tema_ultra = gr.themes.Base(
    font=[gr.themes.GoogleFont("Outfit"), gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#070707", body_background_fill_dark="#070707", body_text_color="#F5F5F7", body_text_color_dark="#F5F5F7", 
    background_fill_primary="#0D0D0D", background_fill_primary_dark="#0D0D0D", background_fill_secondary="#111111", background_fill_secondary_dark="#111111",
    border_color_primary="#2A2A2A", border_color_primary_dark="#2A2A2A", block_background_fill="#0D0D0D", block_background_fill_dark="#0D0D0D",
    block_label_text_color="#C5A059", block_label_text_color_dark="#C5A059", block_title_text_color="#D4AF37", block_title_text_color_dark="#D4AF37", 
    input_background_fill="#121212", input_background_fill_dark="#121212", input_border_color="#333333", input_border_color_dark="#333333",
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

table, .table-wrap, .wrap-inner, .table-container { background-color: #121212 !important; border: 1px solid #333333 !important; border-radius: 8px !important; border-collapse: collapse !important; }
thead th, th { background-color: #0A0A0A !important; color: #D4AF37 !important; border: 1px solid #333333 !important; font-family: 'Outfit', sans-serif !important; letter-spacing: 1px; padding: 12px !important; }
tbody td, td { background-color: #121212 !important; color: #F5F5F7 !important; border: 1px solid #222 !important; padding: 10px !important; }
tbody tr:hover td { background-color: #1A1A1A !important; }
.cell-wrap input { color: #F5F5F7 !important; background-color: transparent !important; }

/* TRAVA DE SCROLL PARA TEXTO DO CHAT */
.chat-container textarea { max-height: 120px !important; overflow-y: auto !important; }

code, pre { background-color: #181818 !important; color: #D4AF37 !important; border: 1px solid #2A2A2A !important; border-radius: 6px !important; }
input:-webkit-autofill { -webkit-box-shadow: 0 0 0 30px #121212 inset !important; -webkit-text-fill-color: #F5F5F7 !important; }
textarea, input, select, .wrap-inner, .dropdown-menu, .wrap { background-color: #121212 !important; color: #F5F5F7 !important; border: 1px solid #333333 !important; border-radius: 8px !important; font-family: 'Inter', sans-serif !important; }
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
        status_cerebro = "🔵 **Modo Matrix Gigante Ativo**" if os.environ.get("OPENROUTER_API_KEY") else f"🟢 **Radar de Córtex Ativo**"
        gr.HTML(f"<h3 style='text-align: center; color: #555; text-transform: uppercase; font-size: 11px; letter-spacing: 4px; margin-bottom: 20px;'>{status_cerebro}</h3>")
        
        with gr.Accordion("⚙️ NÚCLEO DE OPERAÇÕES", open=False):
            gr.Markdown("*Configuração do Comportamento da Máquina.*")
            persona_box = gr.Dropdown(choices=["Assessor Gold (Padrão)", "Estrategista de Vendas", "Auditor de Negócios", "Diretor Criativo"], value="Assessor Gold (Padrão)", show_label=False)
            net_box = gr.Checkbox(label="🌐 Conexão Web (Pesquisa Externa)", value=False)
            btn_limpa = gr.Button("🧹 Purgar Memória Volátil", variant="secondary")
            msg_sys = gr.Textbox(show_label=False, interactive=False)
            btn_limpa.click(fn=limpar_banco_de_dados, outputs=msg_sys)
        
        with gr.Accordion("📡 TERMINAL DE TRANSMISSÃO (WEBHOOKS)", open=False):
            gr.Markdown("*Ponte de injeção de dados direta via Zapier, Make e CRMs Corporativos.*")
            url_webhook = gr.Textbox(placeholder="Insira a Rota do Webhook (URL)...", show_label=False)
            texto_export = gr.Textbox(placeholder="Pacote de Dados a ser enviado...", lines=2, show_label=False)
            btn_web = gr.Button("Disparar Transmissão Externa", variant="primary")
            msg_web = gr.Textbox(show_label=False, interactive=False)
            btn_web.click(fn=disparar_webhook, inputs=[url_webhook, texto_export], outputs=msg_web)

        gr.HTML("<hr style='border: none; border-bottom: 1px solid #1F1F1F; margin: 30px 0;'>")
        btn_back = gr.Button("📦 EXPORTAR BACKUP DO COFRE", variant="primary")
        msg_b = gr.Textbox(show_label=False)
        arq_b = gr.File(label="Arquivo Compactado", visible=False)
        btn_back.click(fn=gerar_backup, outputs=[arq_b, msg_b]).then(lambda: gr.update(visible=True), None, arq_b)

    with gr.Tabs():
        
        with gr.TabItem("🧠 TERMINAL MASTER"):
            chat = gr.ChatInterface(
                fn=responder_chat_multimodal, multimodal=True, additional_inputs=[persona_box, net_box],
                chatbot=gr.Chatbot(height="70vh", show_label=False, placeholder="SISTEMA OMNI ATIVO. QUAL A DIRETRIZ DE COMANDO?"),
                # Trava Gradio ativada: Máximo de 5 linhas antes de criar scroll interno
                textbox=gr.MultimodalTextbox(placeholder="Insira textos táticos, bases de dados Excel, PDFs ou peça para gerar Imagens e Documentos...", container=False, scale=7, show_label=False, max_lines=5)
            )

        with gr.TabItem("🗄️ CAIXA PRETA (LOGS)"):
            with gr.Column(elem_classes="box-painel"):
                gr.Markdown("### 🗄️ LIVRO-CAIXA E HISTÓRICO VITALÍCIO\n*Acesso pericial aos registros de memória da Inteligência Artificial. Atualize para puxar as últimas operações criptografadas.*")
                btn_atualizar_db = gr.Button("ATUALIZAR VISÃO PERICIAL", variant="primary")
                tabela_memoria = gr.Dataframe(headers=["ID", "DATA", "MÓDULO", "REGISTRO DE OPERAÇÃO"], interactive=False, wrap=True)
                btn_atualizar_db.click(fn=ler_memoria, outputs=tabela_memoria)

        with gr.TabItem("🤖 AGENTE DE ELITE"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    gr.Markdown("### 👑 PROTOCOLO DE EXECUÇÃO AUTÔNOMA\n*Delegue uma missão de ponta a ponta. O Agente fará varreduras e pesquisas de mercado em tempo real, elaborará a matriz estratégica e renderizará todos os ativos visuais para a sua campanha.*")
                    txt_missao = gr.Textbox(label="Diretriz da Missão", lines=5, placeholder="Ex: Desenvolva uma estrutura de vendas para o novo empreendimento imobiliário de luxo em São Paulo...")
                    btn_agente = gr.Button("INICIAR PROTOCOLO ALFA", variant="primary", size="lg")
                with gr.Column(scale=6):
                    out_estrat = gr.Textbox(label="Estratégia Sintetizada pelo Agente", lines=16, interactive=False)
                    out_arte = gr.Image(label="Ativo Visual Comercial Renderizado", type="filepath")
            btn_agente.click(fn=executar_agente_mestre, inputs=[txt_missao], outputs=[out_estrat, out_arte])

        with gr.TabItem("📑 TRIBUNAL DE AUDITORIA"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    gr.Markdown("### ⚖️ VARREDURA DE DADOS EM MASSA\n*O Inquisidor Neural cruza dados extensos e emite laudos periciais.*")
                    arq_up = gr.File(label="Submeta o Acervo de Documentos (PDF/Imagens/Excel)", file_count="multiple")
                    txt_ordem = gr.Textbox(label="Diretriz Inquisitória (O que a IA deve auditar?)", lines=3)
                    with gr.Row():
                        c_img = gr.Checkbox(label="🖼️ Renderizar Capa Executiva", value=False)
                        c_aud = gr.Checkbox(label="🔊 Emitir Laudo em Áudio", value=False)
                        c_trib = gr.Checkbox(label="⚖️ Acionar Malha Fina Antifraude", value=False)
                    btn_exe = gr.Button("INICIAR INQUÉRITO DE DADOS", variant="primary")
                with gr.Column(scale=6):
                    out_tela = gr.Textbox(label="Espelho do Laudo", lines=20, interactive=False)
                    with gr.Row():
                        out_word = gr.File(label="Extração Final (Word)")
                        out_aud = gr.Audio(label="Reproduzir Perícia Vocal")
                    out_tel = gr.Textbox(show_label=False, lines=1, interactive=False)
            btn_exe.click(fn=gerar_dossie, inputs=[arq_up, txt_ordem, c_img, c_aud, c_trib], outputs=[msg_sys, out_word, out_aud, out_tela, out_tel])

        with gr.TabItem("🎬 ESTÚDIO DE SÍNTESE"):
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37;'>📸 SÍNTESE FOTOGRÁFICA (8K)</h3><p style='color: #888; font-size: 12px; margin-top: -5px; margin-bottom: 10px;'>Geração de ativos visuais de alto impacto comercial.</p>")
                    img_sujeito = gr.Textbox(label="Foco Central da Direção de Arte")
                    img_fundo = gr.Textbox(label="Construção de Atmosfera (Fundo)")
                    img_estilo = gr.Dropdown(choices=["Fotorrealista 8k", "Cinematic Dark", "Estúdio Minimalista", "Luxo Corporativo"], label="Filtro Estético", value="Fotorrealista 8k")
                    btn_gerar_img = gr.Button("RENDERIZAR COMPOSIÇÃO VISUAL", variant="primary")
                    out_img_est = gr.Image(label="Matriz Extraída", type="filepath")
                    btn_gerar_img.click(fn=gerar_imagem_estudio, inputs=[img_sujeito, img_fundo, img_estilo], outputs=[out_img_est])
                
                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37;'>🎥 DIREÇÃO DE CINEMA IA</h3><p style='color: #888; font-size: 12px; margin-top: -5px; margin-bottom: 10px;'>Animação neural e síntese de movimento em alta definição.</p>")
                    vid_base = gr.Image(label="Base Estrutural (Apenas se quiser animar foto existente)", type="filepath")
                    vid_acao = gr.Textbox(label="Diretriz de Animação ou Roteiro Visual")
                    vid_fundo = gr.Textbox(label="Ambientação do Cenário")
                    vid_mov = gr.Dropdown(choices=["Zoom In Lento (Dramático)", "Afastamento (Zoom Out)", "Movimento Panorâmico", "Captura Aérea (Drone)"], label="Movimento de Câmera Virtual", value="Zoom In Lento (Dramático)")
                    btn_gerar_vid = gr.Button("INICIAR MASTERIZAÇÃO EM VÍDEO", variant="primary")
                    out_vid = gr.Video(label="Arquivo Máster de Saída (MP4)")
                    msg_vid = gr.Textbox(show_label=False, interactive=False)
                    btn_gerar_vid.click(fn=gerar_video_ia, inputs=[vid_base, vid_acao, vid_fundo, vid_mov], outputs=[out_vid, msg_vid])
            
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.HTML("<h3 style='color: #D4AF37;'>🎙️ ESTÚDIO DE LOCUÇÃO NEURAL</h3><p style='color: #888; font-size: 12px; margin-top: -5px; margin-bottom: 10px;'>Clonagem e síntese de voz perfeitamente humana para narrativas.</p>")
                    txt_aud = gr.Textbox(show_label=False, placeholder="Cole o Script de Vendas ou Roteiro Executivo para locução...", lines=5)
                    btn_gerar_aud = gr.Button("SINTETIZAR E MASTERIZAR ÁUDIO", variant="primary")
                    out_aud_estudio = gr.Audio(label="Faixa Consolidada (MP3)")
                    btn_gerar_aud.click(fn=falar_laudo_estudio, inputs=[txt_aud], outputs=[out_aud_estudio])

lista_de_usuarios = []
for i in ["", "_1", "_2", "_3", "_4", "_5"]:
    u = os.environ.get(f"LOGIN_USUARIO{i}") or os.environ.get(f"USUARIO{i}")
    s = os.environ.get(f"LOGIN_SENHA{i}") or os.environ.get(f"SENHA{i}")
    if u and s: lista_de_usuarios.append((u, s))

html_tela_login = renderizar_logo_login() if lista_de_usuarios else None

interface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 10000)), 
    auth=lista_de_usuarios if lista_de_usuarios else None, 
    auth_message=html_tela_login,
    allowed_paths=[os.path.abspath(DIRETORIO)]
)
