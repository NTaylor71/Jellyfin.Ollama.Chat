Jellyfin.Ollama.Chat - Developer Guide
======================================

🚀 Onboarding for New Developers
--------------------------------

1. Clone the repo:

    ```
    git clone https://github.com/NTaylor71/Jellyfin.Ollama.Chat.git
    cd Jellyfin.Ollama.Chat
    ```

2. Copy and configure environment:

    ```
    cp .env.example .env
    ```
    Edit with your model name, collection, ports, etc.


3. Install Python dependencies in a local `.venv`:

    ### Windows
    ```
    ./dev_setup.ps1
    ```

    ### macOS/Linux
    ```bash
    ./dev_setup.sh
    ```


4. Start the full stack:

    ```
    ./build.ps1            # Windows
    ./build.sh             # macOS/Linux
    ```


5. Test ingestion + chat:

    ```
    python scripts/dev.py test-ingest-query
    ```


🧠 What This Project Does
--------------------------
1. Accepts Jellyfin metadata via POST /api/ingest
2. Converts each item to a richly formatted semantic block using Jinja2
3. Embeds using Ollama + LangChain
4. Stores in Qdrant for semantic vector search
5. Accepts chat queries via POST /chat, queues them in Redis
6. Asynchronously processes jobs via worker, returning result via GET /chat/result/{job_id}


💻 Developer Tips
------------------

- Run full test suite:

    ```
    pytest
    ```

- Run individual tests manually:

    ```
    python tests/test_chat.py
    python tests/test_ingest_and_query.py
    ```

- Use the CLI dev tool:

    ```
    python scripts/dev.py cli
    ```

- Other CLI actions:

    ```
    python scripts/dev.py test-chat
    python scripts/dev.py test-ingest-query
    ```

- Manually ingest a batch via API:

    ```
    curl -X POST http://localhost:8000/api/ingest \
      -H "Content-Type: application/json" \
      -d '{"replace": true, "entries": [ ... ]}'
    ```

- Submit a chat query manually:

    ```
    curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"query": "sci-fi movies about simulation"}'
    ```

- Poll for result:

    ```
    curl http://localhost:8000/chat/result/<job_id>
    ```

- Run just the worker manually:

    ```
    docker compose -f docker-compose.dev.yml run --rm worker
    ```

🛠 Troubleshooting
-------------------

- Qdrant is unhealthy:
    
    - Wait for startup, or inspect: docker logs jellychat_vectordb


- Ollama not responding:
    
    - Ensure the model is downloaded and GPU/CPU is available


- Redis not processing jobs:
    
    - Confirm Redis and worker are both running, and polling from chat:queue


- No result after sending a job:
    
    - Use: python scripts/dev.py test-chat to simulate a known-working roundtrip


- Reset persistent data (Qdrant, Ollama models, etc.):

    - docker compose -f docker-compose.dev.yml down -v


✅ Requirements
----------------

- Docker + Docker Compose
- GPU optional (Ollama will fall back to CPU if not available)
- Python 3.13+ only required for running dev_setup.ps1, tests/, and CLI tools


🫶 Contributing
----------------

Pull requests welcome!  
Please include clear commit messages and test your changes locally via:

    pytest
    python scripts/dev.py test-ingest-query
