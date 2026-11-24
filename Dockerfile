FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
# torch CPU par défaut ; pour un GPU : --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data/lexique_corrections.tsv data/memoire_traduction.tsv ./data/

RUN useradd -m -u 1000 wvc && mkdir -p /data /models && chown -R wvc /data /models /srv
USER wvc

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
