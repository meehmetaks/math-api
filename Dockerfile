FROM python:3.10-slim

WORKDIR /app

# OpenCV ve sistem kütüphaneleri
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render'ın dinamik port ataması için ${PORT} kullanımı kritiktir
CMD ["sh", "-c", "uvicorn math_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
