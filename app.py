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
from datetime import datetime
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

# ==========================================
# 1. CHAVES MESTRES E CONEXÕES EM NUVEM
# ==========================================
chave_groq = os.environ.get("GROQ_API_KEY")
chave_hf = os.environ.get("HF_TOKEN")

cliente_groq = Groq(api_key=chave_groq)
cliente_hf = InferenceClient(token=chave_hf)
MODELO_GROQ = "llama-3.3-70b-versatile"

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=chave_hf, 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# 2. DIRETÓRIOS E LOGS
# ==========================================
DIRETORIO = "./Central_IA_Master"
DIR_CHROMA = f"{DIRETORIO}/Banco_de_Dados_Vetorial"
DIR_CASOS = f"{DIRETORIO}/Casos_Periciais"
DIR_MIDIA = f"{DIRETORIO}/Midia_Avulsa"

for d in [DIRETORIO, DIR_CASOS, DIR_MIDIA]:
    os.makedirs(d, exist_ok=True)

def registrar_log(acao):
    agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(f"{DIRETORIO}/Log_Auditoria.txt", "a", encoding="utf-8") as f: 
        f.write(f"[{agora}] - {acao}\n")

def listar_arquivos_mortos():
    arquivos = []
    for d in [DIR_CASOS, DIR_MIDIA]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(('.docx', '.pdf', '.mp3', '.jpg')): arquivos.append(os.path.join(root, f))
    arquivos.sort(key=os.path.getmtime, reverse=True)
    return arquivos

registrar_log("Sistema Minimalista V6.1 Iniciado.")

# ==========================================
# 3. EXTRAÇÃO E FUNÇÕES BASE
# ==========================================
def limpar_banco_de_dados():
    try:
        if os.path.exists(DIR_CHROMA): shutil.rmtree(DIR_CHROMA)
        registrar_log("Banco de Dados Limpo.")
        return "🧹 Mesa Limpa! Memória isolada."
    except Exception as e: return f"Erro: {e}"

def ouvir_microfone(audio_path):
    if not audio_path: return ""
    try:
        with open(audio_path, "rb") as file:
            return cliente_groq.audio.transcriptions.create(file=(audio_path, file.read()), model="whisper-large-v3").text
    except Exception as e: return f"Erro no microfone: {e}"

def extrair_texto(arquivo):
    nome = arquivo.name.lower()
    texto = ""
    try:
        if nome.endswith('.pdf'):
            with pdfplumber.open(arquivo.name) as pdf:
                for pagina in pdf.pages:
                    txt = pagina.extract_text()
                    if txt: texto += txt + "\n"
        elif nome.endswith(('.xlsx', '.csv')):
            texto = (pd.read_excel(arquivo.name) if nome.endswith('.xlsx') else pd.read_csv(arquivo.name)).to_string() 
        elif nome.endswith('.docx'):
            for p in docx.Document(arquivo.name).paragraphs: texto += p.text + "\n"
        elif nome.endswith(('.mp3', '.ogg', '.wav')):
            with open(arquivo.name, "rb") as file:
                texto = f"[TRANSCRIÇÃO]: {cliente_groq.audio.transcriptions.create(file=(arquivo.name, file.read()), model='whisper-large-v3').text}\n"
        return texto
    except: return ""

def falar_laudo(texto_laudo, pasta_destino):
    if not texto_laudo: return None
    cam_txt, cam_audio = f"{pasta_destino}/temp.txt", f"{pasta_destino}/Laudo_Falado.mp3"
    with open(cam_txt, "w", encoding="utf-8") as f: f.write(texto_laudo[:3000].replace('*', ''))
    os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{cam_txt}" --write-media "{cam_audio}"')
    return cam_audio

# ==========================================
# 4. CHAT INTELIGENTE (MOTOR PRINCIPAL)
# ==========================================
def responder_chat(mensagem, historico):
    mensagens = [{"role": "system", "content": "Você é o Assistente Forense Chefe. Comporte-se como uma IA avançada estilo ChatGPT, mas com especialidade em direito, perícia e análise de dados. Responda de forma clara, técnica e objetiva."}]
    for user_txt, ai_txt in historico:
        mensagens.append({"role": "user", "content": user_txt})
        mensagens.append({"role": "assistant", "content": ai_txt})
    mensagens.append({"role": "user", "content": mensagem})
    
    resposta = cliente_groq.chat.completions.create(messages=mensagens, model=MODELO_GROQ, max_tokens=2500).choices[0].message.content
    return resposta

# ==========================================
# 5. DOSSIÊ COMPLEXO
# ==========================================
def gerar_dossie(arquivos, instrucao, usar_img, usar_aud, usar_tribunal, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Faltou instrução", None, None, "", ""
    t_inicio = time.time()
    palavras = 0
    try:
        progresso(0.1, desc="Iniciando Operação...")
        pasta = f"{DIR_CASOS}/Caso_{datetime.now().strftime('%d_%m_%Y__%Hh%M')}"
        os.makedirs(pasta, exist_ok=True)
        
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for idx, arq in enumerate(arquivos):
                progresso((0.1 + (0.3 * (idx/len(arquivos)))), desc="Processando Evidências...")
                txt = extrair_texto(arq)
                palavras += len(txt.split())
                chunks = fatiador.split_text(txt)
                banco.add_texts([f"[FONTE: {os.path.basename(arq.name)}]\n{c}" for c in chunks])
            
        progresso(0.5, desc="Cruzando os dados...")
        contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        
        prompt = f"""Você é um Perito Sênior. 
        CRÍTICO: Cite as fontes lidas entre colchetes.
        DADOS: {contexto}
        AÇÃO: {instrucao}"""
        
        resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ, max_tokens=4000).choices[0].message.content
        
        progresso(0.8, desc="Gerando Relatório...")
        cam_word = f"{pasta}/Laudo.docx"
        doc = docx.Document()
        doc.add_heading('Laudo Pericial Oficial', 0)
        doc.add_paragraph(resposta)
        doc.save(cam_word)
        
        cam_audio = falar_laudo("Resumo: " + resposta[:1000], pasta) if usar_aud else None
        telemetria = f"📊 STATUS: {len(arquivos) if arquivos else 0} arquivos | {palavras} palavras | {round(time.time() - t_inicio, 1)}s"
        
        progresso(1.0, desc="Pronto!")
        return "✅ Concluído", cam_word, cam_audio, resposta, telemetria
    except Exception as e: return f"Erro: {e}", None, None, "", ""

def gerar_backup():
    cam = "./Backup_Central.zip"
    with zipfile.ZipFile(cam, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(DIRETORIO):
            if "Banco_de_Dados_Vetorial" not in root:
                for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), DIRETORIO))
    return cam, "📦 Backup Pronto"

# ==========================================
# 6. INTERFACE MINIMALISTA (TIPO CHATGPT)
# ==========================================
tema_chatgpt = gr.themes.Default(
    primary_hue="zinc",
    secondary_hue="zinc",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)
css_minimalista = """
footer {display: none !important;} 
.gradio-container {border-top: none; max-width: 1200px !important;}
.tabs {border-bottom: 1px solid #e5e7eb;}
.box-painel {border: 1px solid #e5e7eb; padding: 20px; border-radius: 8px; background-color: #ffffff;}
"""

with gr.Blocks(title="Assistente Forense", theme=tema_chatgpt, css=css_minimalista) as interface:
    
    with gr.Tabs():
        
        # ABA 1 AGORA É O CHAT (PÁGINA INICIAL)
        with gr.TabItem("💬 Assistente Forense"):
            chat = gr.ChatInterface(
                fn=responder_chat,
                chatbot=gr.Chatbot(height=650, placeholder="Olá, Diretor. Como posso ajudar na investigação hoje?"),
                textbox=gr.Textbox(placeholder="Envie uma mensagem para a IA...", container=False, scale=7)
            )

        # ABA 2 É O COCKPIT PARA DOCUMENTOS PESADOS
        with gr.TabItem("🔎 Processador de Documentos (Dossiês)"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    gr.Markdown("### 📂 Inserir Lote de Evidências")
                    arq_up = gr.File(label="PDFs, Excel, Word, Áudios", file_count="multiple")
                    
                    gr.Markdown("### 🎯 Ordem de Serviço")
                    with gr.Row():
                        mic = gr.Audio(sources=["microphone"], type="filepath", label="Falar")
                        txt_ordem = gr.Textbox(show_label=False, lines=4, placeholder="Descreva o que a IA deve buscar ou analisar nestes arquivos...")
                    mic.stop_recording(fn=ouvir_microfone, inputs=mic, outputs=txt_ordem)
                    
                    with gr.Row():
                        c_aud = gr.Checkbox(label="🔊 Gerar Laudo em Áudio", value=False)
                        c_trib = gr.Checkbox(label="⚖️ Modo Debate (Tribunal)", value=False)
                    
                    btn_exe = gr.Button("Gerar Dossiê Oficial", variant="primary", size="lg")
                    btn_limpa = gr.Button("Limpar Memória do Caso", variant="secondary")
                    msg_sys = gr.Textbox(show_label=False, interactive=False)
                    btn_limpa.click(fn=limpar_banco_de_dados, outputs=msg_sys)

                with gr.Column(scale=6):
                    out_tela = gr.Textbox(label="Laudo Executivo", lines=20, interactive=False)
                    with gr.Row():
                        out_word = gr.File(label="Baixar Word")
                        out_aud = gr.Audio(label="Ouvir Resumo")
                    out_tel = gr.Textbox(show_label=False, lines=1, interactive=False)
                    
            btn_exe.click(fn=gerar_dossie, inputs=[arq_up, txt_ordem, c_aud, c_aud, c_trib], outputs=[msg_sys, out_word, out_aud, out_tela, out_tel])

        # ABA 3: ARQUIVOS
        with gr.TabItem("🗄️ Arquivos & Backup"):
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### Histórico de Casos")
                    btn_lista = gr.Button("Atualizar Lista")
                    galeria = gr.File(label="Documentos Gerados", file_count="multiple", interactive=False)
                    btn_lista.click(fn=listar_arquivos_mortos, outputs=galeria)
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("### Segurança")
                    btn_back = gr.Button("Gerar Backup Completo", variant="primary")
                    msg_b = gr.Textbox(show_label=False)
                    arq_b = gr.File(label="Arquivo ZIP")
                    btn_back.click(fn=gerar_backup, outputs=[arq_b, msg_b])

# ==========================================
# 7. GERENCIAMENTO SEGURO DE ACESSOS 
# ==========================================
lista_de_usuarios = []

if os.environ.get("LOGIN_USUARIO") and os.environ.get("LOGIN_SENHA"):
    lista_de_usuarios.append((os.environ.get("LOGIN_USUARIO"), os.environ.get("LOGIN_SENHA")))
if os.environ.get("USUARIO_1") and os.environ.get("SENHA_1"):
    lista_de_usuarios.append((os.environ.get("USUARIO_1"), os.environ.get("SENHA_1")))
if os.environ.get("USUARIO_2") and os.environ.get("SENHA_2"):
    lista_de_usuarios.append((os.environ.get("USUARIO_2"), os.environ.get("SENHA_2")))
if os.environ.get("USUARIO_3") and os.environ.get("SENHA_3"):
    lista_de_usuarios.append((os.environ.get("USUARIO_3"), os.environ.get("SENHA_3")))

interface.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 10000)), 
    auth=lista_de_usuarios
)
