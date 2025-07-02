# 📦 Final Project Plan — FAISS RAG System with Redis Queue, Hot Plugins, and Seamless Local/Docker Switching

## ✅ Core Design Summary

### 🔌 Dedicated FAISS service for vector CRUD API.
### ⚙️ Redis-backed async queue for GPU task processing.
### 🔁 API with hot-reloadable plugins supporting weighted execution and live file detection.
### ♻️ Safe restart via bootstrap.py in API only.
### 🩺 Uniform healthcheck system using FastAPI endpoints and curl-based Docker healthchecks.
### 🛠 Precise 1:1 folder mapping between /docker/ and /src/.
### 🚀 Seamless onboarding and local development using dev_setup.ps1 and build.ps1 with dynamic .env and Docker environment switching.
### ✅ Locked service names:

- redis_queue_worker
- faiss_ingestor
- faiss_service

## 📂 Repository Structure

```
.env.example
.gitignore
build.ps1
build.sh
dev_setup.ps1
dev_setup.sh
docker-compose.dev.yml
pyproject.toml
README.md
docker\wait_for_services.sh
docker\api\Dockerfile
docker\api\entrypoint.sh
docker\api\healthcheck.sh
docker\faiss_ingestor\Dockerfile
docker\faiss_ingestor\entrypoint.sh
docker\faiss_ingestor\healthcheck.sh
docker\faiss_service\Dockerfile
docker\faiss_service\entrypoint.sh
docker\faiss_service\healthcheck.sh
docker\ingestor\Dockerfile.dev
docker\ingestor\entrypoint.sh
docker\ollama\entrypoint.sh
docker\ollama\healthcheck.sh
docker\redis_queue_worker\Dockerfile
docker\redis_queue_worker\entrypoint.sh
docker\redis_queue_worker\healthcheck.sh
docker\redis_queue_worker\main.py
docker\vectordb\Dockerfile
docker\vectordb\healthcheck.sh
docker\worker\Dockerfile.dev
docker\worker\entrypoint.sh
scripts\cli_send.py
scripts\dev.py
src\config.py
src\__init__.py
src\api\bootstrap.py
src\api\entrypoint.py
src\api\faiss_client.py
src\api\healthcheck.py
src\api\main.py
src\api\plugin_loader.py
src\api\plugin_registry.py
src\api\safe_restart.py
src\api\server.py
src\api\__init__.py
src\faiss_ingestor\entrypoint.py
src\faiss_ingestor\healthcheck.py
src\faiss_ingestor\ingestion_worker.py
src\faiss_ingestor\main.py
src\faiss_ingestor\__init__.py
src\faiss_service\entrypoint.py
src\faiss_service\faiss_index.py
src\faiss_service\faiss_server.py
src\faiss_service\healthcheck.py
src\faiss_service\main.py
src\faiss_service\worker.py
src\faiss_service\__init__.py
src\ollama_service\entrypoint.py
src\ollama_service\healthcheck.py
src\ollama_service\ingestion_worker.py
src\ollama_service\main.py
src\ollama_service\__init__.py
src\plugins\embed_data_embellisher_starter.py
src\plugins\faiss_crud_plugin_starter.py
src\plugins\query_embellisher_starter.py
src\plugins\__init__.py
src\rag\formatter.py
src\rag\media_template.j2
src\rag\__init__.py
src\redis_queue_worker\main.py
src\redis_queue_worker\New Text Document.txt
src\redis_queue_worker\__init__.py
src\tests\test_chat.py
src\tests\test_end_to_end.py
src\tests\test_faiss_ingestor.py
src\tests\test_faiss_service.py
src\tests\test_formatter.py
src\tests\test_ingest.py
src\tests\test_ingest_and_query.py
src\tests\test_plugins.py
src\tests\test_redis_queue_worker.py
src\tests\__init__.py
```
## ✅ Plugin Types and Execution Points

| Plugin Type              | Description                                   | Called From               |
|--------------------------|-----------------------------------------------|---------------------------|
| Query Embellishers       | Modify or augment the user’s chat query        | API (pre-queue enqueue)   |
| Embed Data Embellishers  | Modify or augment embedding documents          | FAISS Ingestor            |
| FAISS CRUD Plugins       | Apply additional logic to FAISS CRUD events    | FAISS Service             |

## 🔑 Plugin Types and Contexts encore

| Plugin Type              | Execution Context         | Purpose                                    | Trigger Point                                   |
|--------------------------|---------------------------|--------------------------------------------|------------------------------------------------|
| 1. Query Embellishers    | API Container             | Modify or enhance the user's query.        | Called before enqueuing to Redis queue.        |
| 2. Embed Data Embellishers| FAISS Ingestor Container  | Modify or enhance embedding documents.     | Called during ingestion, before sending to FAISS service. |
| 3. FAISS CRUD Plugins    | FAISS Service Container   | Inject custom logic into FAISS CRUD ops.   | Called on add, search, or delete.              |

---

## ✔️ Execution Path Examples:

- Query embellishers: 

  - API → modifies incoming request before sending to Redis queue.

- Embed data embellishers: 

  - Ingestor → modifies documents before sending to FAISS Service.

- FAISS CRUD plugins:
    - FAISS Service → augments CRUD behavior on add/search/delete.

## ✅ Seamless Local/Docker Environment Design
### Dev Environment:

- .env.example → used by local dev scripts.

- config.py → detects Docker vs. local via ENV=docker switch.

- dev_setup.ps1 → loads local .env and runs Docker scripts seamlessly.

- build.ps1 → builds all Docker containers with correct env bindings.


### Docker Environment:

- Docker Compose uses container-scoped environment variables like:

    - yaml
        ```
        environment:
          - ENV=docker
          - REDIS_HOST=redis
          - OLLAMA_BASE_URL=http://jellychat_ollama:11434
          - VECTORDB_URL=http://faiss_service:6333
        ```
- Containers automatically resolve Redis, Ollama, and FAISS using Docker network aliases.

### Config.py Auto-Switch Example:

```commandline
import os
from dotenv import load_dotenv

if os.getenv("ENV") != "docker":
    load_dotenv()

IS_DOCKER = os.getenv("ENV") == "docker"

REDIS_HOST = "redis" if IS_DOCKER else os.getenv("REDIS_HOST", "localhost")
OLLAMA_BASE_URL = "http://jellychat_ollama:11434" if IS_DOCKER else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VECTORDB_URL = "http://faiss_service:6333" if IS_DOCKER else os.getenv("VECTORDB_URL", "http://localhost:6333")
```





## ✅ Phase 1 Deliverables
- All Dockerfiles

- All entrypoint.sh files

- All healthcheck.sh files

- Shared wait_for_services.sh

- docker-compose.dev.yml with healthcheck dependencies and volume mappings

- Local .env.example with correctly mapped variables

- dev_setup.ps1 and build.ps1 fully configured to switch between local and Docker environments automatically


## ✅ Detailed Execution Flow
### 1. 🔧 Query Embellishers
- 📦 Container: API

- 🎯 Purpose:

    - Enrich, sanitize, or reformat the user’s input query.

    - Add extra metadata, tags, or rephrase queries for better search precision.

- 🗂 Example Use:

    - Auto-append genre filters, auto-detect date ranges, normalize titles.

- ✅ When Called:

    - API → before pushing the request to the Redis queue.

- 🔁 Ordering:

    - Plugins are executed in ascending weight order.

- 🛑 Failure Handling:

    - A failure in embellishment aborts the request with a clean API error.

### 2. 🔧 Embed Data Embellishers
= 📦 Container: FAISS Ingestor

- 🎯 Purpose:

    - Format, enrich, or clean metadata before FAISS indexing.

    - Example: add extra fields, apply Jinja2 templates, remove noisy text.

- ✅ When Called:

    - During ingestion → just before sending the FAISS add request.

- 🔁 Ordering:

    - Plugins are executed in ascending weight order.

- 🛑 Failure Handling:

    - If embellishment fails → skip the document or abort ingestion gracefully.

### 3. 🔧 FAISS CRUD Plugins

- 📦 Container: FAISS Service

- 🎯 Purpose:
    - Apply additional logic during FAISS add, search, and delete operations.
    = Example: log indexing stats, conditionally block deletes, inject monitoring.

- ✅ When Called:
  - On every add, search, or delete FAISS operation.

- 🔁 Ordering:

  - Plugins are executed in ascending weight order.

- 🛑 Failure Handling:

    - If a plugin fails → log, skip the plugin, but proceed with CRUD operation unless a hard block is configured.

- ✅ Plugin System Control
  - All plugins must:

    - Declare their type: Query Embellisher, Embed Data Embellisher, or FAISS CRUD Plugin.
    - Declare their weight: Used to control execution order.
    - Register themselves in the plugin registry.

  - API has admin endpoints to:
    - List loaded plugins.
    - Manually trigger plugin reloads.
    - Safe restart the API process if a reload fails.



## System Prompt:
"The user is building a Docker-first FAISS RAG system with Redis-backed async queue processing and a dedicated FAISS service. The API supports hot-reloadable plugins with weighted execution order and uses a safe restart mechanism via bootstrap.py. Redis queue workers handle LLM tasks with Ollama. The FAISS Ingestor is a separate container that populates the vector store. All containers must have uniform healthchecks and use a shared wait_for_services.sh script. The user expects seamless onboarding using dev_setup.ps1 and build.ps1, which automatically switch environment variables between local dev and Docker using a unified config.py. The user strongly prefers precise folder mappings between /docker/ and /src/ and expects complete, file-level answers with no missing components. All design decisions should prioritize minimal, consistent, scalable Docker architecture, fast onboarding, and local dev simplicity."

## ✅ Phase Plan encore

| Phase | Deliverables                                                      |
|-------|-------------------------------------------------------------------|
| 1     | Dockerfiles, entrypoints, healthcheck.sh files, wait_for_services.sh, docker-compose.dev.yml |
| 2     | API container: FastAPI, plugin loader, bootstrap, safe restart    |
| 3     | FAISS Service: CRUD API, index management                         |
| 4     | Redis + RQ queue integration                                      |
| 5     | Redis Queue Worker: Redis queue listener, Ollama processor        |
| 6     | FAISS Ingestor: ingestion pipeline, FAISS updates                 |
| 7     | Plugin hot-reload system                                          |
| 8     | Full safe restart system                                          |
| 9     | Full testing suite, dev scripts, system polish                    |


# Next steps

## COMPLETED :

### STAGE 1 : build repo skeleton

- we step through the proposed files, one by one

  - you ask me for possibly existing examples of each file from a previous repo
    - i reply with a few examples to keep you on track
        - you trandlate to this project
    - OR, I reply 'thats new, no examples'
      - you create the code 

  - at all stages I want full code replacement, never snippets
  - at all steps I want you to indicate the full path of the file being referenced 
  - any zips i give you are to be unpacked with python, a file map is created, then all files in the map are read so you understand ALL the contents of the provided zip

- strict rule : we will step through all the files from the project-root/ in struct alphabetical order

### COMPLETED : STAGE 2 : debugging the build

### CURRENT STAGE : STAGE 3 : testing the test script test_redis_queue_worker.py

- help me debug any issues when we run python test_redis_queue_worker.py



## 🔍 LLM Movie Search Strategy Overview

| Layer                        | Purpose                                       | Your Status         | Next Step                        |
|-----------------------------|-----------------------------------------------|----------------------|----------------------------------|
| 1. Prompt Enrichment        | Expand vague query into rich semantic anchor  | ✅ Working (via LLM) | 🎯 Tune inflation + extract keywords |
| 2. Data Enrichment          | Rich movie blobs with thematic metadata       | ✅ Done (via LLM)    | ➕ Add per-section embedding        |
| 3. Embedding + Vector Store | Represent concepts as search vectors          | ✅ FAISS + NumPy     | ⚖️ Optimize per-section weights     |
| 4. Scoring Logic            | Compute final match quality                   | ✅ Cosine now        | 🧪 Add blended or hybrid scoring    |
| 5. Literal Fields (future)  | Guarantee high-precision filters              | ❌ Not yet           | 🧠 Plan boosts using tags/genres    |





