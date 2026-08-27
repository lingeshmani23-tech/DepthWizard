# DepthWizard AI Computation Backend Dockerfile
FROM python:3.10-slim

# Install system dependencies for OpenCV and OpenGL rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend dependencies
COPY requirements.txt .

# Install Python ML/AI stack
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code and config
COPY . /app/backend
COPY src /app/src
COPY config.py /app/config.py
COPY sample_images /app/sample_images

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
