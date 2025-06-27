import asyncio
from src.faiss_ingestor.ingestion_worker import start_ingestion

def main():
    print("🚀 FAISS Ingestor starting ingestion loop...")
    asyncio.run(start_ingestion())

if __name__ == "__main__":
    main()
