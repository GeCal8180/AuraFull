import gradio as gr
import pandas as pd
import os
import docx
from docx.shared import Inches
import io
import base64
import pdfplumber
import pytesseract
import shutil
import re
import zipfile
import time
from datetime import datetime
from pdf2image import convert_from_path
from PIL import Image
from fpdf import FPDF
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from huggingface_hub import InferenceClient

# ==========================================
# 1. CHAVES MESTRES E CONEXÕES EM NUVEM
# ==========================================
# As chaves reais não ficam mais aqui. O Render vai injetar elas com segurança.
chave_groq = os.environ.get("GROQ_API_KEY")
chave_hf = os.environ.get("HF_TOKEN")

cliente_groq = Groq(api_key=chave_groq)
cliente_hf = InferenceClient(token=chave_hf)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
MODELO_GROQ = "llama-3.3-70b-versatile"
MODELO_VISAO = "llama-3.2-90b-vision-preview"
buscador_web = DuckDuckGoSearchRun()

# ==========================================
# 2. DIRETÓRIOS DO SERVIDOR (NUVEM)
# ==========================================
DIRETORIO = "./Central_IA_Master"
DIR_CHROMA = f"{DIRETORIO}/Banco_de_Dados_Vetorial"
DIR_CASOS = f"{DIRETORIO}/Casos_Periciais"
DIR_MIDIA = f"{DIRETORIO}/Midia_Avulsa"

for d in [DIRETORIO, DIR_CASOS, DIR_MIDIA]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 3. LOG E ARQUIVO MORTO
# ==========================================
def registrar_log(acao):
    agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    caminho_log = f"{DIRETORIO}/Log_Auditoria_Forense.txt"
    with open(caminho_log, "a", encoding="utf-8") as f: f.write(f"[{agora}] - {acao}\n")
    return caminho_log

def listar_arquivos_mortos():
    arquivos = []
    for d in [DIR_CASOS, DIR_MIDIA]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(('.docx', '.pdf', '.mp3', '.jpg', '.mp4', '.xlsx')):
                        arquivos.append(os.path.join(root, f))
    arquivos.sort(key=os.path.getmtime, reverse=True)
    return arquivos

registrar_log("Servidor em Nuvem 24/7 (SaaS) Iniciado com Sucesso.")

# ==========================================
# 4. LIMPEZA E EXTRAÇÃO (COM WHISPER NA NUVEM)
# ==========================================
def limpar_banco_de_dados():
    try:
        if os.path.exists(DIR_CHROMA): shutil.rmtree(DIR_CHROMA)
        registrar_log("Banco de Dados Limpo (Isolamento de Caso).")
        return "🧹 Mesa Limpa! Sistema isolado e seguro para o próximo caso."
    except Exception as e: return f"Erro: {e}"

def ouvir_microfone_diretor(audio_path):
    """Agora usa a API da Groq: 100x mais rápido e sem peso no servidor"""
    if not audio_path: return ""
    try:
        with open(audio_path, "rb") as file:
            transcricao = cliente_groq.audio.transcriptions.create(file=(audio_path, file.read()), model="whisper-large-v3")
        return transcricao.text
    except Exception as e: return f"Erro no microfone: {e}"

def extrair_texto(arquivo):
    nome = arquivo.name.lower()
    texto = ""
    try:
        if nome.endswith('.pdf'):
            with pdfplumber.open(arquivo.name) as pdf:
                for i, pagina in enumerate(pdf.pages):
                    txt_pagina = pagina.extract_text()
                    if txt_pagina and len(txt_pagina.strip()) > 20: texto += txt_pagina + "\n"
                    else:
                        imagens = convert_from_path(arquivo.name, first_page=i+1, last_page=i+1)
                        for img in imagens: texto += pytesseract.image_to_string(img, lang='por') + "\n"
        elif nome.endswith(('.xlsx', '.csv')):
            texto = (pd.read_excel(arquivo.name) if nome.endswith('.xlsx') else pd.read_csv(arquivo.name)).to_string() 
        elif nome.endswith('.docx'):
            for p in docx.Document(arquivo.name).paragraphs: texto += p.text + "\n"
        elif nome.endswith(('.mp3', '.ogg', '.wav', '.m4a')):
            # Áudio terceirizado para a Nuvem da Groq
            with open(arquivo.name, "rb") as file:
                texto = f"[TRANSCRIÇÃO DE ÁUDIO]: {cliente_groq.audio.transcriptions.create(file=(arquivo.name, file.read()), model='whisper-large-v3').text}\n"
        return texto
    except Exception as e: return f"Erro na extração: {e}"

def falar_laudo_caso(texto_laudo, pasta_destino):
    if not texto_laudo: return None
    cam_txt = f"{pasta_destino}/temp_texto.txt"
    cam_audio = f"{pasta_destino}/Laudo_Falado.mp3"
    texto_limpo = texto_laudo.replace('"', '').replace('🚨', '').replace('*', '')[:3000]
    with open(cam_txt, "w", encoding="utf-8") as f: f.write(texto_limpo)
    os.system(f'edge-tts --voice pt-BR-AntonioNeural -f "{cam_txt}" --write-media "{cam_audio}"')
    return cam_audio

# ==========================================
# 5. O MOTOR DOSSIÊ COM PASTAS E CITAÇÕES
# ==========================================
def gerar_dossie_completo(arquivos, instrucao, usar_imagem, usar_audio, usar_tribunal, progresso=gr.Progress()):
    if not instrucao: return "⚠️ Faltou instrução", None, None, "Aguardando ordem...", ""
    tempo_inicio = time.time()
    palavras_totais = 0
    
    try:
        progresso(0.1, desc="Iniciando a Central SaaS...")
        nome_pasta_caso = f"Caso_{datetime.now().strftime('%d_%m_%Y__%Hh%M')}"
        pasta_caso = f"{DIR_CASOS}/{nome_pasta_caso}"
        os.makedirs(pasta_caso, exist_ok=True)
        
        banco = Chroma(persist_directory=DIR_CHROMA, embedding_function=embeddings)
        fatiador = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        if arquivos:
            total_arquivos = len(arquivos)
            for idx, arq in enumerate(arquivos):
                nome_arq = os.path.basename(arq.name)
                progresso((0.1 + (0.3 * (idx/total_arquivos))), desc=f"Lendo: {nome_arq[:20]}...")
                texto_arq = extrair_texto(arq)
                palavras_totais += len(texto_arq.split())
                chunks = fatiador.split_text(texto_arq)
                chunks_com_fonte = [f"[FONTE DO TRECHO: {nome_arq}]\n{c}" for c in chunks]
                banco.add_texts(chunks_com_fonte)
            
        progresso(0.5, desc="Buscando evidências...")
        contexto = "\n\n".join([doc.page_content for doc in banco.similarity_search(instrucao, k=10)])
        
        progresso(0.6, desc="Tribunal de IA em sessão..." if usar_tribunal else "Llama 3 analisando...")
        regra_imagem = "\nNo final, pule uma linha e escreva: [IMAGEM: descreva em INGLÊS cena fotorrealista para capa]" if usar_imagem else ""
        regra_citacao = "CRÍTICO: Ao apresentar qualquer dado, valor ou contradição, você DEVE Citar a Fonte colocando o nome do arquivo lido entre colchetes. Exemplo: 'O valor cobrado foi R$ 10.000 [FONTE DO TRECHO: contrato.pdf]'."
        
        if usar_tribunal:
            prompt_mestre = f"""Você é um Tribunal Forense de IA. Siga 3 passos:
            1. VISÃO DO PROMOTOR: Ataque os dados e aponte crimes/falhas.
            2. VISÃO DA DEFESA: Justifique e defenda os dados.
            3. VEREDITO DO PERITO (LAUDO): Sintetize as visões em um laudo neutro e blindado.
            {regra_citacao} {regra_imagem}
            DADOS: {contexto}
            AÇÃO: {instrucao}"""
        else:
            prompt_mestre = f"""Você é um Perito Sênior. Faça um laudo executivo detalhado. Use tabelas textuais se necessário.
            {regra_citacao} {regra_imagem}
            DADOS: {contexto}
            AÇÃO: {instrucao}"""
        
        resposta_bruta = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": prompt_mestre}], model=MODELO_GROQ, max_tokens=4000).choices[0].message.content
        texto_laudo = resposta_bruta
        cam_img = None

        if usar_imagem:
            progresso(0.7, desc="Gerando capa ilustrativa (HF Cloud)...")
            match = re.search(r'\[IMAGEM:\s*(.*?)\]', resposta_bruta, re.IGNORECASE)
            texto_laudo = re.sub(r'\[IMAGEM:\s*(.*?)\]', '', resposta_bruta, flags=re.IGNORECASE).strip()
            cam_img = f"{pasta_caso}/Capa_Dossie.jpg"
            cliente_hf.text_to_image(match.group(1).strip() if match else "Forensic dossier", model="black-forest-labs/FLUX.1-schnell").save(cam_img)
        
        progresso(0.8, desc="Formatando Word e Pastas...")
        cam_word = f"{pasta_caso}/Laudo_Pericial.docx"
        doc = docx.Document()
        doc.add_heading('Laudo Pericial Oficial', 0)
        if cam_img: doc.add_picture(cam_img, width=Inches(6.0))
        doc.add_paragraph(texto_laudo)
        doc.save(cam_word)
        registrar_log(f"Laudo gerado e salvo na pasta: {nome_pasta_caso}")
        
        cam_audio = falar_laudo_caso("Resumo: " + texto_laudo[:1000], pasta_caso) if usar_audio else None
        
        tempo_total = round(time.time() - tempo_inicio, 1)
        telemetria = f"📊 STATUS DA OPERAÇÃO (NUVEM):\nArquivos Analisados: {len(arquivos) if arquivos else 0}\nVolume Lido: {palavras_totais} palavras\nTempo Total: {tempo_total} segundos\nSalvo em: /{nome_pasta_caso}"
        
        progresso(1.0, desc="Concluído!")
        return "✅ Dossiê Gerado!", cam_word, cam_audio, texto_laudo, telemetria
    except Exception as e: return f"Erro: {e}", None, None, "Falha.", f"Erro: {e}"

def gerar_backup_sistema():
    registrar_log("Backup Completo solicitado.")
    caminho_zip = "./Backup_Central_Forense.zip"
    try:
        with zipfile.ZipFile(caminho_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(DIRETORIO):
                if "Banco_de_Dados_Vetorial" in root: continue 
                for file in files: zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), DIRETORIO))
        return caminho_zip, "📦 Backup gerado!"
    except Exception as e: return None, f"Erro: {e}"

# ==========================================
# 6. ESTÚDIO VISUAL (APIs TERCERIZADAS)
# ==========================================
def chat_memoria(mensagem, historico, usar_web):
    contexto = f"[Web: {buscador_web.run(mensagem)}]\n" if usar_web else ""
    return cliente_groq.chat.completions.create(messages=[{"role": "user", "content": contexto + mensagem}], model=MODELO_GROQ, max_tokens=2000).choices[0].message.content

def gerar_imagem(p):
    c = f"{DIR_MIDIA}/Img_{datetime.now().strftime('%H%M%S')}.jpg"
    try: p_eng = cliente_groq.chat.completions.create(messages=[{"role": "user", "content": f"Translate to English: {p}"}], model=MODELO_GROQ).choices[0].message.content.replace('"', '').strip()
    except: p_eng = p
    cliente_hf.text_to_image(p_eng, model="black-forest-labs/FLUX.1-schnell").save(c)
    return c

# ==========================================
# 7. A INTERFACE NUVEM V5.0
# ==========================================
tema_premium = gr.themes.Soft(primary_hue="slate", secondary_hue="slate")
menu_templates = {
    "": "",
    "🎯 Laudo Executivo": "Faça uma varredura geral e monte um laudo completo, cruzando valores e criando tabelas textuais.",
    "⚖️ Acareação de Provas": "Realize uma acareação rigorosa entre os arquivos fornecidos. Aponte divergências."
}
css_corporativo = "footer {display: none !important;} .gradio-container {border-top: 8px solid #334155; border-radius: 5px;}"

with gr.Blocks(theme=tema_premium, css=css_corporativo, title="Central IA Master") as interface:
    gr.Markdown("# 🏛️ Sistema Central de Inteligência Forense (SaaS Cloud)")
    
    with gr.Tabs():
        with gr.TabItem("🔎 Cockpit de Análise"):
            with gr.Row():
                with gr.Column(scale=4):
                    arq_upload = gr.File(label="📄 Lote de Arquivos (PDFs, Excel, Áudios)", file_count="multiple")
                    gr.Markdown("**Dite ou Digite a Instrução:**")
                    with gr.Row():
                        mic_ordem = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Falar Ordem")
                        txt_ordem = gr.Textbox(show_label=False, lines=4, placeholder="Sua instrução aparecerá aqui...")
                    mic_ordem.stop_recording(fn=ouvir_microfone_diretor, inputs=mic_ordem, outputs=txt_ordem)
                    
                    with gr.Accordion("⚙️ Configurações Avançadas", open=False):
                        drop_templates = gr.Dropdown(choices=list(menu_templates.keys()), label="Atalhos Rápidos")
                        drop_templates.change(fn=lambda e: menu_templates[e], inputs=drop_templates, outputs=txt_ordem)
                        chk_tribunal = gr.Checkbox(label="⚖️ Ativar Tribunal (Debate entre Promotor e Defesa)", value=False)
                        chk_img = gr.Checkbox(label="🖼️ Gerar Capa Ilustrativa", value=False)
                        chk_aud = gr.Checkbox(label="🔊 Gerar Parecer em Áudio", value=False)
                    
                    btn_dossie = gr.Button("🚀 EXECUTAR PERÍCIA", variant="primary", size="lg")
                    btn_limpar = gr.Button("🧹 Isolar Caso (Zerar Memória)", variant="stop")
                    msg_limpeza = gr.Textbox(show_label=False, interactive=False)
                    btn_limpar.click(fn=limpar_banco_de_dados, inputs=[], outputs=msg_limpeza)

                with gr.Column(scale=6):
                    saida_msg = gr.Textbox(show_label=False, interactive=False, placeholder="Aguardando comando...")
                    with gr.Row():
                        arq_dossie = gr.File(label="📄 Baixar Dossiê (Word)")
                        audio_dossie = gr.Audio(label="🔊 Ouvir Laudo")
                    painel_telemetria = gr.Textbox(label="Painel de Telemetria (Raio-X de Dados)", lines=4, interactive=False)
                    saida_texto_visor = gr.Textbox(label="Pré-visualização do Laudo na Tela", lines=15, interactive=False)
                
            btn_dossie.click(fn=gerar_dossie_completo, inputs=[arq_upload, txt_ordem, chk_img, chk_aud, chk_tribunal], outputs=[saida_msg, arq_dossie, audio_dossie, saida_texto_visor, painel_telemetria])

        with gr.TabItem("🗄️ Arquivo Morto"):
            btn_atualizar_morto = gr.Button("🔄 Atualizar Lista", variant="primary")
            galeria_arquivos = gr.File(label="Clique para baixar os arquivos antigos", file_count="multiple", interactive=False)
            btn_atualizar_morto.click(fn=listar_arquivos_mortos, inputs=[], outputs=galeria_arquivos)

        with gr.TabItem("🔒 Segurança & Logs"):
            gr.Markdown("⚠️ **Aviso de Nuvem:** Servidores gratuitos reiniciam após horas de inatividade. Faça este Backup no fim do seu expediente para garantir que não perderá os arquivos!")
            btn_backup = gr.Button("⬇️ GERAR BACKUP COMPLETO DO EXPEDIENTE (.ZIP)", variant="primary")
            msg_backup = gr.Textbox(show_label=False, interactive=False)
            arq_backup = gr.File(label="Arquivo ZIP")
            visor_log = gr.Textbox(label="Diário de Bordo Forense", lines=10, interactive=False)
            
            def atualizar_backup():
                log = ""
                try: 
                    with open(f"{DIRETORIO}/Log_Auditoria_Forense.txt", "r") as f: log = f.read()[-3000:]
                except: log = "Log vazio."
                bck_path, bck_msg = gerar_backup_sistema()
                return bck_path, bck_msg, log
            btn_backup.click(fn=atualizar_backup, inputs=[], outputs=[arq_backup, msg_backup, visor_log])

        with gr.TabItem("🎬 Estúdio Avulso"):
            gr.Markdown("*Nota: O módulo de vídeos pesados foi extraído para manter a gratuidade e a leveza deste painel 24/7. Use este espaço para gerar imagens de alta resolução táticas.*")
            with gr.Row():
                with gr.Column():
                    txt_midia = gr.Textbox(label="Descreva a cena:")
                    b_img = gr.Button("Gerar Imagem", variant="primary")
                with gr.Column():
                    out_img = gr.Image(label="Imagem Final", type="filepath")
            b_img.click(fn=gerar_imagem, inputs=[txt_midia], outputs=[out_img])

# Puxando o usuário e senha do cofre do Render
usuario_seguro = os.environ.get("LOGIN_USUARIO")
senha_segura = os.environ.get("LOGIN_SENHA")

# Lançamento seguro e blindado
interface.launch(server_name="0.0.0.0", server_port=10000, auth=(usuario_seguro, senha_segura))
