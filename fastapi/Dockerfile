# Gunakan image Python yang ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies yang dibutuhkan oleh librosa/soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements terlebih dahulu untuk memanfaatkan caching layer Docker
COPY requirements.txt .

# Install dependencies Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code ke dalam image
COPY . .

# Expose port yang digunakan (sesuai main.py)
EXPOSE 8001

# Jalankan aplikasi
CMD ["python", "main.py"]
