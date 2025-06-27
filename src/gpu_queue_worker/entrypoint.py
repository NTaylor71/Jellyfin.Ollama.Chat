import uvicorn
from src.ollama_service.healthcheck import app

def main():
    print("🚀 Ollama Service starting healthcheck server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
