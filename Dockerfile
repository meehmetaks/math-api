# Temel imaj
FROM python:3.9-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları (OpenCV için gerekli)
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Uygulamayı başlat
CMD ["uvicorn", "math_server:app", "--host", "0.0.0.0", "--port", "8000"]
