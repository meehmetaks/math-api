# Tam Python sürümü
FROM python:3.10

WORKDIR /app

# Sistem kütüphaneleri (OpenCV için gerekli)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Kütüphaneleri requirements.txt ile yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Uygulamayı başlat
CMD ["uvicorn", "math_server:app", "--host", "0.0.0.0", "--port", "8000"]
