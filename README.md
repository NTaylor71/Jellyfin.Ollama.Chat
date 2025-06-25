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

```
Jellyfin.Ollama.Chat/
├── .env                     # Active environment config (copied from .env.example)
├── .env.example             # Sample env values (Postgres, Ollama, etc.)
├── .gitignore               # Ignores Python cache, Docker volumes, secrets
├── build.ps1                # PowerShell: builds + runs the dev stack
├── build.sh                 # Bash equivalent of build.ps1
├── docker-compose.dev.yml  # Dev environment definition
├── manage.py                # Django launcher (uses webserver.settings)
├── pyproject.toml           # PEP 621 + hatch config
├── README.md                # This file

├── docker/
│   ├── ingestor/
│   │   ├── Dockerfile.dev       # Ingestor container (LangChain + vector upload)
│   │   └── entrypoint.sh        # Waits for Qdrant/Ollama, then runs ingest
│   ├── vectordb/
│   │   ├── Dockerfile           # Extends qdrant/qdrant to include a healthcheck script
│   │   └── healthcheck.sh       # Robust startup check for Qdrant
│   ├── web/
│   │   ├── Dockerfile.dev       # Django container
│   │   └── entrypoint.sh        # Waits for DB, runs migrate + runserver
│   └── worker/
│       ├── Dockerfile.dev       # RAG query worker (LangChain + Ollama)
│       └── entrypoint.sh        # Waits for dependencies, then runs LangChain

├── src/
│   ├── ingestor/
│   │   └── main.py              # Embeds Jellyfin metadata into Qdrant
│   ├── webserver/
│   │   ├── __init__.py
│   │   ├── asgi.py
│   │   ├── settings.py          # Django settings (DJANGO_SETTINGS_MODULE=webserver.settings)
│   │   ├── urls.py
│   │   └── wsgi.py
│   └── worker/
│       └── main.py              # RAG query handler using LangChain + Ollama

```

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
