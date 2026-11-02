# Imagem da API de churn. Construir e rodar:
#   docker build -t churn-api .
#   docker run --rm -p 8000:8000 churn-api

# ----- Etapa 1: instala só as dependências de produção, nas versões do uv.lock -----
FROM python:3.14-slim AS dependencias

RUN pip install --no-cache-dir uv==0.13.6
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN UV_PYTHON_DOWNLOADS=never uv sync --frozen --no-dev --no-cache

# ----- Etapa 2: imagem final, sem o uv e sem os arquivos de instalação -----
FROM python:3.14-slim

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY --from=dependencias /app/.venv .venv
COPY src src
COPY models/02_modelo_campeao.joblib models/02_modelo_campeao_metadata.json models/

# A API não precisa escrever em disco, então roda sem privilégios de root
USER nobody

# As nuvens costumam informar a porta na variável PORT (o Cloud Run usa 8080)
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "import os, urllib.request as u; u.urlopen(f\"http://127.0.0.1:{os.getenv('PORT', '8000')}/health\", timeout=4)"]

CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
