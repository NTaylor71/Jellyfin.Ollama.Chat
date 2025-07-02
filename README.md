# 🚀 Production RAG System

Professional FAISS RAG system with Redis queue processing and hot-reloadable plugins.

## Quick Start

1. **Setup**: `./dev_setup.ps1`
2. **Test**: `python -m pytest`
3. **Develop**: Edit code and enjoy!

## Key Features

- 🔍 FAISS vector search
- ⚡ Redis queue processing  
- 🔌 Hot-reloadable plugins
- 🐳 Docker-first design
- 📊 Production monitoring

## Architecture

```
API ──► Redis Queue ──► FAISS Service
 │                          │
 └──► Plugin System         └──► Vector Search
```

## Development

```bash
# Start development environment
./dev_setup.ps1

# Run tests
python -m pytest

# Check configuration
python -c "from src.shared.config import get_settings; print(get_settings())"
```

## Configuration

Key settings in `.env`:

- `ENV`: localhost, docker, or production
- `REDIS_HOST`: Redis server location
- `OLLAMA_CHAT_BASE_URL`: Chat model endpoint
- `VECTORDB_URL`: FAISS service endpoint

## Next Steps

This is a foundational setup. Additional components will be added incrementally:

1. ✅ Basic configuration and structure
2. 🔄 FastAPI application
3. 🔄 FAISS service
4. 🔄 Redis worker
5. 🔄 Plugin system
6. 🔄 Docker containers
7. 🔄 Monitoring

---

Happy coding! 🚀
