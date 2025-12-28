# Temel imaj
FROM python:3.9-bullseye

# Çalışma klasörü
WORKDIR /app

# Sistem paketlerini yükle
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ca-certificates \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Sunucuyu başlat
CMD ["uvicorn", "math_server:app", "--host", "0.0.0.0", "--port", "8000"]
