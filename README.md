# Jellyfin.Ollama.Chat

Local-first AI chatbot for your Jellyfin library.  
Powered by Django, LangChain, Ollama, and Qdrant.

---

## 🚀 Quick Start (for Developers)

### 1. Clone the Repo

    git clone https://github.com/YOUR_ORG/Jellyfin.Ollama.Chat.git
    cd Jellyfin.Ollama.Chat

### 2. Setup `.env`

    cp .env.example .env

Edit the `.env` file if needed (e.g. change database password or Ollama model).

---

### 3. Build and Launch

**Windows (PowerShell):**

    ./build.ps1

**Linux/macOS (Bash):**

    ./build.sh

This will:
- Build Docker containers for Django, LangChain worker, ingestor, PostgreSQL, Ollama, and Qdrant
- Launch everything in dev mode

---

## 🧪 Services

    Django (API/admin):   http://localhost:8000
    Qdrant UI:            http://localhost:6333
    Ollama (LLM):         http://localhost:12434
    Postgres:             localhost:5432

---

## 🧱 Project Layout

Jellyfin.Ollama.Chat/
├── .env                     # Active environment config (copied from .env.example)
├── .env.example            # Sample configuration (PostgreSQL, Ollama, etc.)
├── .gitignore              # Excludes volumes, Python cache, .env, etc.
├── build.ps1               # PowerShell build & launch script
├── build.sh                # Bash equivalent of build.ps1
├── docker-compose.dev.yml # Full dev stack: Django, Ollama, Qdrant, etc.
├── manage.py               # Django launcher (DJANGO_SETTINGS_MODULE points to webserver.settings)
├── pyproject.toml          # Project config (PEP 621, hatchling)
├── README.md               # Dev-friendly instructions and architecture overview

├── docker/                         # All container-related logic
│   ├── ingestor/
│   │   ├── Dockerfile.dev         # Builds the ingestor container from src/ingestor
│   │   └── entrypoint.sh          # Waits for Qdrant, Ollama, then runs LangChain ingestion
│   ├── vectordb/
│   │   ├── Dockerfile             # Extends qdrant/qdrant, adds custom healthcheck.sh
│   │   └── healthcheck.sh         # Robust check for Qdrant readiness
│   ├── web/
│   │   ├── Dockerfile.dev         # Django app container (builds from pyproject + src/)
│   │   └── entrypoint.sh          # Waits for PostgreSQL, runs migrations, starts dev server
│   └── worker/
│       ├── Dockerfile.dev         # LangChain query container
│       └── entrypoint.sh          # Waits for Ollama, Qdrant, PostgreSQL, then runs LangChain

├── src/                            # All source code lives here
│   ├── ingestor/
│   │   └── main.py                # Embeds Jellyfin data into Qdrant using LangChain
│   ├── webserver/
│   │   ├── __init__.py           # Marks this as a Python module
│   │   ├── asgi.py               # ASGI entrypoint
│   │   ├── settings.py           # Django config (updated to use 'webserver' module path)
│   │   ├── urls.py               # Routing
│   │   └── wsgi.py               # WSGI entrypoint
│   └── worker/
│       └── main.py              # LangChain RAG query execution using Ollama


---

## 🧠 What This Project Does

1. Ingests your Jellyfin media metadata into a vector DB (Qdrant)
2. Embeds it using Ollama + LangChain
3. Supports querying your media collection via natural language
4. Runs 100% locally with GPU acceleration (if supported)

---

## 💻 Developer Tips

- Run just the ingest step:

      docker compose run --rm ingestor

- Run just the query worker:

      docker compose run --rm worker

- Inspect the database:

      docker exec -it jellychat_db psql -U chatdb

---

## 🛠 Troubleshooting

- Qdrant is unhealthy:
      wait a few seconds or inspect the healthcheck log
- Django crash: `ModuleNotFoundError`
      → make sure DJANGO_SETTINGS_MODULE is set to `"webserver.settings"`
- Database errors:
      → ensure POSTGRES_DB in `.env` is `chatdb`
- Reset persistent state:
      docker compose down -v

---

## ✅ Requirements

- Docker + Docker Compose
- GPU optional (Ollama will fall back to CPU)
- Python 3.12+ required only if developing outside Docker

---

## 🫶 Contributing

Pull requests welcome!  
Please include clear commit messages and test your changes locally.
