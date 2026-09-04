FROM python:3.10-slim

# Instala dependências do sistema necessárias para áudio/vídeo e banco de dados
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PORT=8080

CMD ["python", "app.py"]
