
# FAISS RAG System - Developer Guide

## 🎉 Welcome

This project is proudly built to **welcome junior and intermediate developers.**

You are not expected to know everything. We explain every tool and every step.

---

## 📅 Key Tools and Concepts

- **Docker**: Used to run everything consistently.
- **Redis**: Handles queues for async processing.
- **FAISS**: Fast Approximate Nearest Neighbor search.
- **RAG (Retrieval Augmented Generation)**: Uses document search to improve LLM responses.
- **Jinja2**: Used to format data into clean text for the AI.
- **Environment Variables**: Easily switch between local and Docker setups.

---

## 💡 What This Project Does

- Accepts metadata from Jellyfin.
- Formats metadata with Jinja2.
- Sends vectors to a FAISS database.
- Processes chat queries through a Redis queue.
- Returns search results quickly.
- Supports **hot-reloadable plugins** for customization.

---

## ✨ Step-by-Step Developer Setup

### 1. Clone the Repo

```bash
git clone https://github.com/SGloop/faiss-rag-system.git
cd faiss-rag-system
```

### 2. Setup Environment

```bash
cp .env.example .env
```

- Edit `.env` if needed (model names, Redis URLs, etc.).
  - Editing is only needed in professional deployments, it is fine in your LAN at home as is.

### 3. Build Local Python Environment

#### Windows

```powershell
./dev_setup.ps1
```

#### macOS/Linux

```bash
./dev_setup.sh
```

### 4. Start Full System

#### Windows

```powershell
./build.ps1
```

#### macOS/Linux

```bash
./build.sh
```

---

## 🔍 How the System Works

- Chat queries go to the **Redis queue.**
- The worker pulls jobs from the queue.
- Results are pushed back to Redis.
- The FAISS database handles all vector searches.
- Plugins can modify:
  - User queries
  - Documents sent to FAISS for turning into RAG data
  - FAISS CRUD operations

---

## 👨‍💼 Dev Tools and Commands

### CLI Dev Tool

```bash
python scripts/dev.py cli
```

### Test Full Roundtrip

```bash
python scripts/dev.py test-chat
```

### Test Ingest + Query

```bash
python scripts/dev.py test-ingest-query
```

### Run Individual Tests

```bash
pytest
python tests/test_chat.py
python tests/test_faiss_service.py
```

---

## 📂 Environment Switching

The `ENV` variable in `.env` tells the system whether to run in Docker or local mode.

- Docker: `ENV=docker`
- Local: `ENV=local`

The system will **auto-detect** and adjust Redis hosts, ports, and URLs accordingly.

- No need to change this if just running at home within your LAN.

ENV=local

- Just means Python that you run on your host machine (PC) can access the internal Docker network that you build via Docker.

ENV=docker

- Used by 'docker build', all URLs use the internal Docker environment for URLs.

In practice, you can ignore this concept and everything env var 'just works'.

---

## 🔍 Understanding RAG (Retrieval Augmented Generation)

**What is RAG?**

RAG (Retrieval Augmented Generation) is a system where we **search documents first**, then pass the results into an AI prompt to improve its answers.

Instead of asking the AI to "know everything," we:
1. Store helpful documents (movie metadata, synopses, tags, etc.).
2. Find relevant documents **at query time** using FAISS.
3. Send the retrieved documents into the AI prompt to provide **better, more accurate, and grounded answers.**

This system is:
- 🏎️ Fast because FAISS uses vector-based nearest neighbor search.
- 🔍 Focused because results come from your database, not random AI guesses.

---

## 🗅️ Understanding FAISS (Facebook AI Similarity Search)

**What is FAISS?**

FAISS is a **vector database** that finds the most similar entries to a given query vector.

### Core Concepts:
- Each movie/document is **converted into a vector** (a long list of numbers).
- FAISS can quickly find **the nearest vectors** (most similar items) to a search query.
- FAISS stores **both the vector and the associated metadata.**

### ⚙️ How Persistence Works
- By default, FAISS stores the index **in memory.**
- FAISS can also **save to disk** to persist between restarts.
- This project uses FAISS with optional disk persistence via `faiss_index.idx`.

If the index file is missing and rebuilding is not allowed, the system will fail fast. This is intentional so developers can choose whether to:
- 🚀 Allow automatic rebuild.
- 🚩 Require manual rebuild.

### ➕ Growing the FAISS Database
- You can add **one movie at a time** or batch ingest multiple movies.
- Each ingestion step sends a formatted document to the FAISS Service.
- You can grow the database continuously — new content can be added anytime.

### ↻ Execution Lifecycle

1. **Ingestion**
   - Metadata is converted to text using **Jinja2 templates.**
   - The text is embedded (converted to vectors).
   - Vectors and metadata are sent to FAISS via the ingestion pipeline.

2. **Chat Query**
   - The user sends a chat query via the API.
   - The system pushes the query to a Redis queue.
   - The worker picks up the job.

3. **Search**
   - The worker queries FAISS to find the most relevant documents.
   - FAISS returns the closest matches.

4. **Result Delivery**
   - The worker sends the search results back to Redis.
   - The API returns the results to the user.

### 🔌 How Plugins and Templates Fit In

#### 📦 Query Embellishers
- Run **before** the chat job is pushed to Redis.
- Example: Automatically append "genre:sci-fi" if the user types "space."

#### 🧹 Embed Data Embellishers
- Run **during ingestion** to format or enrich documents.
- Example: Add "Language: English" if missing.

#### 🛠️ FAISS CRUD Plugins
- Run **during add, search, or delete operations** in the FAISS Service.
- Example: Log vector stats, modify vector data before indexing.

#### 🖋️ Templates
- Jinja2 templates shape each document into a **clean, LLM-friendly format.**
- The more readable and detailed the template, the better the AI can use the data.

### 🚀 Key Takeaway for Juniors and Intermediates:
- You can **grow the system incrementally.**
- You can write your own plugins.
- You can experiment with small, isolated test cases.
- You can contribute without knowing advanced math or AI theory.

---

## 📚 Plugin System

### 3 Types of Plugins

| Type                    | Purpose                                  | Runs In       |
| ----------------------- | ---------------------------------------- | ------------- |
| Query Embellishers      | Modify incoming chat queries             | API           |
| Embed Data Embellishers | Modify documents before sending to FAISS | Ingestor      |
| FAISS CRUD Plugins      | Modify FAISS add/search/delete ops       | FAISS Service |

- Plugins are **hot-reloadable** and can live in a mounted volume.
- Each plugin can be weighted to control execution order.

---

## 🔧 Useful CURL Examples

### Submit a Chat Query

```bash
curl -X POST http://localhost:8000/chat  -H "Content-Type: application/json"  -d '{"query": "sci-fi movies about simulation"}'
```

### Check Chat Result

```bash
curl http://localhost:8000/chat/result/<job_id>
```

### Submit a Manual Ingest Job to Redis

```bash
python tests/test_faiss_ingestor.py
```

---

## 🛠️ Troubleshooting

- **FAISS healthcheck failing?**
  - Run: `docker logs <faiss_service_container>`

- **Redis queue stuck?**
  - Run: `python scripts/dev.py test-chat` to simulate a working query.

- **No response after job submission?**
  - Confirm the worker is running and Redis is reachable.

- **Reset system:**
```bash
docker compose -f docker-compose.dev.yml down -v
```

---

## 🧁️ We Welcome Contributions

Pull requests are welcome at all levels.

Please:

- Write clear commit messages.
- Test changes using:

```bash
pytest
python scripts/dev.py test-ingest-query
```

If you are unsure about contributing, reach out — this project is proudly beginner and intermediate friendly.
