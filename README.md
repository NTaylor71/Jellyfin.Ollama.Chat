# Jellyfin.Ollama.Chat

🧠 Local-first AI RAG system using Jellyfin metadata + LangChain + Ollama.

## 🔧 Features

- Django for admin + metadata scraping
- Ingests Jellyfin DB into vector DB (Qdrant)
- Ollama for local LLM
- LangChain-powered RAG with embeddings
- Runs fully offline in Docker

## 🚀 Getting Started

```bash
cp .env.example .env
./build.ps1
```

To ingest Jellyfin data:
```bash
docker compose run --rm ingestor
```

To query:
```bash
docker compose run --rm worker
```
