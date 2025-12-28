# Python 3.10 slim imajı kullanıyoruz, böylece apt-get hataları minimize olur
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları minimal, sadece Pillow ve OpenCV için gerekli
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Uvicorn ile FastAPI başlat
CMD ["uvicorn", "math_server:app", "--host", "0.0.0.0", "--port", "8000"]
