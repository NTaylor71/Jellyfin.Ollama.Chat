import uvicorn
from src.faiss_service.faiss_server import app

def main():
    print("🚀 FAISS Service booting...")
    uvicorn.run(app, host="0.0.0.0", port=6333, reload=False)

if __name__ == "__main__":
    main()
