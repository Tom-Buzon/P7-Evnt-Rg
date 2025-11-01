# Dockerfile
FROM python:3.11-slim

# OS deps utiles (faiss-cpu -> libgomp1 ; build basique)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 gcc curl && rm -rf /var/lib/apt/lists/*

# Installer Poetry
ENV POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Copier les fichiers de dépendances d’abord (pour profiter du cache)
COPY pyproject.toml poetry.lock* ./

# Installer les deps dans l’environnement système du conteneur (pas de venv)
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --only main

# Copier le code et scripts
COPY src ./src
COPY scripts ./scripts
COPY README.md ./

# Dossiers de données/index (persistés via volumes côté docker-compose)
RUN mkdir -p /app/data /app/artifacts/index

# Variables par défaut (à overrider via compose)
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    MISTRAL_API_KEY="" \
    DATA_DIR=/app/data \
    CLEAN_DIR=/app/data/clean \
    INDEX_DIR=/app/artifacts/index

# Exposer l’API FastAPI
EXPOSE 8000

# Démarrer l’API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
