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
from gradio_client import Client  # O Túnel para processamento de Vídeo

# ==========================================
# 1. CHAVES MESTRES E CONEXÕES EM NUVEM
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

def registrar_log(acao):
    agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(f"{DIRETORIO}/Log_Sistema.txt", "a", encoding="utf-8") as f: 
        f.write(f"[{agora}] - {acao}\n")

def listar_arquivos_mortos():
    arquivos = []
    for d in [DIR_CASOS, DIR_MIDIA]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(('.docx', '.pdf', '.mp3', '.jpg', '.mp4')): arquivos.append(os.path.join(root, f))
    arquivos.sort(key=os.path.getmtime, reverse=True)
    return arquivos

# ==========================================
# 3. EXTRAÇÃO DE DADOS
# ==========================================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extrair_texto(arquivo):
    caminho = arquivo.name if hasattr(arquivo, 'name') else arquivo
    nome = caminho.lower()
    texto = ""
    try:
        if nome.endswith('.pdf'):
            with pdfplumber.open(caminho) as pdf:
                for pagina in pdf.pages:
                    txt = pagina.extract_text()
                    if txt: texto += txt + "\n"
        elif nome.endswith(('.xlsx', '.csv')):
            texto = (pd.read_excel(caminho) if nome.endswith('.xlsx') else pd.read_csv(caminho)).to_string() 
        elif nome.endswith('.docx'):
            for p in docx.Document(caminho).paragraphs: texto += p.text + "\n"
        elif nome.endswith(('.mp3', '.ogg', '.wav')):
            with open(caminho, "rb") as file:
                texto = f"[TRANSCRIÇÃO]: {cliente_groq.audio.transcriptions.create(file=(caminho, file.read()), model='whisper-large-v3').text}\n"
        return texto
    except: return ""

def limpar_banco_de_dados():
    try:
        if os.path.exists(DIR_CHROMA): shutil.rmtree(DIR_CHROMA)
        return "🧹 Memória limpa com sucesso."
    except Exception as e: return f"Erro: {e}"

def ouvir_microfone(audio_path):
    if not audio_path: return ""
    try:
        with open(audio_path, "rb") as file:
            return cliente_groq.audio.transcriptions.create(file=(audio_path, file.read()), model="whisper-large-v3").text
    except Exception as e: return f"Erro: {e}"

# ==========================================
# 4. CHAT MULTIMODAL (MARKETING & ANÁLISE)
# ==========================================
def responder_chat_multimodal(mensagem, historico, persona, usar_internet):
    texto_usuario = mensagem.get("text", "")
    arquivos = mensagem.get("files", [])
    contexto_extra, imagens = "", []
    
    for arq in arquivos:
        if arq.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): imagens.append(arq)
        else: contexto_extra += f"\n[DOC ANEXADO]:\n{extrair_texto(arq)}\n"
            
    if usar_internet and texto_usuario:
        try:
            resultados = DDGS().text(texto_usuario, max_results=3)
            contexto_extra += "\n\n[DADOS WEB]:\n" + "\n".join([f"Fonte: {r['title']} - Resumo: {r['body']}" for r in resultados])
        except: pass

    sys_prompt = f"Você é um {persona}. Responda de forma criativa, estratégica ou técnica, adequando-se perfeitamente ao seu papel."
    mensagens = [{"role": "system", "content": sys_prompt}]
    
    for h_u, h_a in historico:
        user_text = "[Mídia Anexada]" if isinstance(h_u, tuple) else str(h_u)
        if h_u: mensagens.append({"role": "user", "content": user_text})
        if h_a: mensagens.append({"role": "assistant", "content": str(h_a)})
    
    texto_final = texto_usuario + contexto_extra

    if imagens:
        conteudo_msg = [{"type": "text", "text": texto_final}]
        for img in imagens:
            conteudo_msg.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"}})
        mensagens.append({"role": "user", "content": conteudo_msg})
        modelo = MODELO_VISAO
    else:
        mensagens.append({"role": "user", "content": texto_final})
        modelo = MODELO_GROQ

    return cliente_groq.chat.completions.create(messages=mensagens, model=modelo, max_tokens=2500).choices[0].message.content

def exportar_conversa(historico):
    if not historico: return None
    pasta = f"{DIR_CASOS}/Chat_{datetime.now().strftime('%d_%m_%H%M')}"
    os.makedirs(pasta, exist_ok=True)
    cam_word = f"{pasta}/Historico_Chat.docx"
    doc = docx.Document()
    doc.add_heading('Relatório da Sessão Criativa/Analítica', 0)
    for h_u, h_a in historico:
        user_text = "[Mídia/Arquivo Enviado]" if isinstance(h_u, tuple) else str(h_u)
        doc.add_heading('Você:', level=2)
        doc.add_paragraph(user_text)
        doc.add_heading('Assistente IA:', level=2)
        doc.add_paragraph(str(h_a))
    doc.save(cam_word)
    return cam_word

# ==========================================
# 5. PROCESSADOR DE DOCUMENTOS LARGOS
# ==========================================
def gerar_dossie(arquivos, instrucao, usar_img, usar_aud, usar_tribunal, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Faltou instrução", None, None, "", ""
    t_inicio = time.time()
    palavras = 0
    try:
        progresso(0.1, desc="Iniciando Processamento...")
        pasta = f"{DIR_CASOS}/Projeto_{datetime.now().strftime('%d_%m_%Y__%Hh%M')}"
        os.makedirs(pasta, exist_ok=True)
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            for idx, arq in enumerate(arquivos):
                progresso((0.1 + (0.3 * (idx/len(arquivos)))), desc="Lendo arquivos...")
                txt = extrair_texto(arq)
                palavras += len(txt.split())
                banco.add_texts([f"[FONTE: {os.path.basename(arq.name)}]\n{c}" for c in fatiador.split_text(txt)])
            
        progresso(0.5, desc="Cruzando dados com a IA...")
        contexto = "\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=8)])
        regra_imagem = "\nNo final do documento, pule uma linha e escreva: [IMAGEM: descreva em INGLÊS uma cena fotorrealista relacionada a este conteúdo]" if usar_img else ""
        prompt = f"Você é um Especialista de Inteligência Sênior. Cite as fontes lidas entre colchetes. {regra_imagem}\nDADOS: {contexto}\nAÇÃO: {instrucao}"
        
        resposta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=MODELO_GROQ, max_tokens=4000).choices[0].message.content
        
        cam_img = None
        if usar_img:
            progresso(0.8, desc="Gerando Arte Ilustrativa...")
            match = re.search(r'\[IMAGEM:\s*(.*?)\]', resposta, re.IGNORECASE)
            resposta_limpa = re.sub(r'\[IMAGEM:\s*(.*?)\]', '', resposta, flags=re.IGNORECASE).strip()
            prompt_img = match.group(1).strip() if match else "High quality professional business illustration"
            cam_img = f"{pasta}/Capa_Projeto.jpg"
            cliente_hf.text_to_image(prompt_img, model="black-forest-labs/FLUX.1-schnell").save(cam_img)
        else:
            resposta_limpa = resposta

        progresso(0.9, desc="Exportando Relatório...")
        cam_word = f"{pasta}/Documento_Final.docx"
        doc = docx.Document()
        doc.add_heading('Relatório Executivo Oficial', 0)
        if cam_img: doc.add_picture(cam_img, width=Inches(6.0))
        doc.add_paragraph(resposta_limpa)
        doc.save(cam_word)
        
        progresso(1.0, desc="Pronto!")
        return "✅ Concluído", cam_word, falar_laudo(resposta_limpa[:1000], pasta) if usar_aud else None, resposta_limpa, f"📊 STATUS: {palavras} palavras processadas."
    except Exception as e: return f"Erro: {e}", None, None, "", ""

# ==========================================
# 6. ESTÚDIO DE CRIAÇÃO (IMAGEM, VÍDEO E ÁUDIO)
# ==========================================
def gerar_imagem_estudio(p):
    if not p: return None
    c = f"{DIR_MIDIA}/Img_{datetime.now().strftime('%H%M%S')}.jpg"
    try: p_eng = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": f"Translate this to English (return only the translation): {p}"}], model=MODELO_GROQ).choices[0].message.content.replace('"', '').strip()
    except: p_eng = p
    cliente_hf.text_to_image(p_eng, model="black-forest-labs/FLUX.1-schnell").save(c)
    return c

def gerar_video_ia(prompt):
    if not prompt: return None, "⚠️ Digite uma cena."
    try:
        # Traduz o roteiro para inglês (melhor precisão no vídeo)
        p_eng = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": f"Translate this scene to English (only translation): {prompt}"}], model=MODELO_GROQ).choices[0].message.content.strip()
        
        # Túnel de API para o supercomputador público da Hugging Face (Text-to-Video)
        cliente_video = Client("multimodalart/zeroscope-v2")
        resultado = cliente_video.predict(p_eng, api_name="/infer")
        
        return resultado, "✅ Vídeo HD gerado com sucesso!"
    except Exception as e:
        return None, f"⚠️ Os supercomputadores públicos de vídeo estão lotados no momento. Tente novamente em alguns minutos."

def falar_laudo_estudio(texto):
    if not texto: return None
    cam_txt, cam_audio = f"{DIR_MIDIA}/temp.txt", f"{DIR_MIDIA}/Voz_{datetime.now().strftime('%H%M%S')}.mp3"
    with open(cam_txt, "w", encoding="utf-8") as f: f.write(texto[:3000].replace('*', ''))
    os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{cam_txt}" --write-media "{cam_audio}"')
    return cam_audio

def gerar_backup():
    cam = "./Backup_Projetos.zip"
    with zipfile.ZipFile(cam, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(DIRETORIO):
            if "Banco_de_Dados_Vetorial" not in root:
                for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), DIRETORIO))
    return cam, "📦 Backup Pronto"

# ==========================================
# 7. INTERFACE UNIVERSAL & TABS
# ==========================================
tema_chatgpt = gr.themes.Default(primary_hue="zinc", secondary_hue="zinc", neutral_hue="zinc", font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
css_min = "footer {display: none !important;} .gradio-container {border-top: none; max-width: 1200px !important;} .box-painel {border: 1px solid #e5e7eb; padding: 20px; border-radius: 8px; background-color: #ffffff;}"

with gr.Blocks(title="Central de Inteligência", theme=tema_chatgpt, css=css_min) as interface:
    gr.Markdown("# 🧠 Central de Inteligência Master")
    gr.Markdown("*Plataforma Universal de Criação, Marketing e Análise Estratégica.*")

    with gr.Tabs():
        
        with gr.TabItem("💬 Assistente Universal (Chat)"):
            with gr.Row():
                btn_exportar = gr.Button("💾 Salvar Roteiro/Conversa (Word)", variant="secondary")
                arq_exportado = gr.File(label="Seu Histórico")
            chat = gr.ChatInterface(
                fn=responder_chat_multimodal, multimodal=True,
                additional_inputs=[
                    gr.Dropdown(choices=["Assistente Universal (Padrão)", "Especialista em Marketing & Copywriting (TikTok, Shopee, Ads)", "Consultor de Negócios & Estratégia", "Auditor Financeiro", "Revisor Jurídico"], value="Assistente Universal (Padrão)", label="🎭 Especialidade da IA"),
                    gr.Checkbox(label="🌐 Permitir Busca na Internet", value=False)
                ],
                chatbot=gr.Chatbot(height=600, placeholder="Peça um roteiro de vídeo, anexe planilhas, imagens ou pesquise na Web..."),
                textbox=gr.Textbox(placeholder="Digite sua ideia ou arraste arquivos...", container=False, scale=7)
            )
            btn_exportar.click(fn=exportar_conversa, inputs=[chat.chatbot], outputs=[arq_exportado])

        with gr.TabItem("🔎 Processador de Documentos"):
            with gr.Row():
                with gr.Column(scale=4, elem_classes="box-painel"):
                    arq_up = gr.File(label="Lote de Documentos (PDF, Excel, Word)", file_count="multiple")
                    txt_ordem = gr.Textbox(label="Instrução Estratégica", lines=3, placeholder="Ex: Crie um resumo executivo deste material...")
                    with gr.Row():
                        c_img = gr.Checkbox(label="🖼️ Capa", value=False)
                        c_aud = gr.Checkbox(label="🔊 Áudio", value=False)
                        c_trib = gr.Checkbox(label="⚖️ Debate", value=False)
                    btn_exe = gr.Button("Processar Lote", variant="primary", size="lg")
                    btn_limpa = gr.Button("Limpar Memória Atual", variant="secondary")
                    msg_sys = gr.Textbox(show_label=False, interactive=False)
                    btn_limpa.click(fn=limpar_banco_de_dados, outputs=msg_sys)
                with gr.Column(scale=6):
                    out_tela = gr.Textbox(label="Visualização do Relatório", lines=20, interactive=False)
                    with gr.Row():
                        out_word = gr.File(label="Baixar Documento Final")
                        out_aud = gr.Audio(label="Ouvir Resumo")
                    out_tel = gr.Textbox(show_label=False, lines=1, interactive=False)
            btn_exe.click(fn=gerar_dossie, inputs=[arq_up, txt_ordem, c_img, c_aud, c_trib], outputs=[msg_sys, out_word, out_aud, out_tela, out_tel])

        with gr.TabItem("🎨 Estúdio de Criação (Mídias)"):
            gr.Markdown("### Geração de Mídia Avançada para Anúncios")
            
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("**🖼️ Criar Imagem (FLUX.1)**")
                    txt_img = gr.Textbox(show_label=False, placeholder="Descreva o produto, cenário ou identidade visual...", lines=2)
                    btn_gerar_img = gr.Button("Gerar Arte / Imagem", variant="primary")
                    out_img = gr.Image(label="Resultado da IA", type="filepath")
                    btn_gerar_img.click(fn=gerar_imagem_estudio, inputs=[txt_img], outputs=[out_img])

                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("**🎬 Criar Vídeo HD (Túnel IA)**")
                    txt_vid = gr.Textbox(show_label=False, placeholder="Descreva a cena animada que deseja gerar...", lines=2)
                    btn_gerar_vid = gr.Button("Gerar Vídeo", variant="primary")
                    out_vid = gr.Video(label="Vídeo Final (MP4)")
                    msg_vid = gr.Textbox(show_label=False, interactive=False)
                    btn_gerar_vid.click(fn=gerar_video_ia, inputs=[txt_vid], outputs=[out_vid, msg_vid])
            
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    gr.Markdown("**🔊 Locução Sintética (Voz)**")
                    txt_aud = gr.Textbox(show_label=False, placeholder="Cole o roteiro do TikTok/Shopee aqui para gerar a voz profissional...", lines=2)
                    btn_gerar_aud = gr.Button("Gerar Locução", variant="primary")
                    out_aud_estudio = gr.Audio(label="Áudio Pronto")
                    btn_gerar_aud.click(fn=falar_laudo_estudio, inputs=[txt_aud], outputs=[out_aud_estudio])

        with gr.TabItem("🗄️ Meus Arquivos & Backup"):
            with gr.Row():
                with gr.Column(elem_classes="box-painel"):
                    btn_lista = gr.Button("Atualizar Lista de Mídias")
                    galeria = gr.File(label="Todos os arquivos gerados nesta sessão", file_count="multiple", interactive=False)
                    btn_lista.click(fn=listar_arquivos_mortos, outputs=galeria)
                with gr.Column(elem_classes="box-painel"):
                    btn_back = gr.Button("Exportar Backup de Segurança (ZIP)", variant="primary")
                    msg_b = gr.Textbox(show_label=False)
                    arq_b = gr.File(label="Download do ZIP")
                    btn_back.click(fn=gerar_backup, outputs=[arq_b, msg_b])

# ==========================================
# 8. ACESSOS SEGUROS NO COFRE
# ==========================================
lista_de_usuarios = []
for i in ["", "_1", "_2", "_3"]:
    u, s = os.environ.get(f"LOGIN_USUARIO{i}" if i=="" else f"USUARIO{i}"), os.environ.get(f"LOGIN_SENHA{i}" if i=="" else f"SENHA{i}")
    if u and s: lista_de_usuarios.append((u, s))

interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)), auth=lista_de_usuarios)
