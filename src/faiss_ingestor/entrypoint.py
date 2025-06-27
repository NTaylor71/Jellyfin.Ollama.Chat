import asyncio
from .ingestion_worker import start_ingestion

def main():
    print("🚀 FAISS Ingestor booting...")
    asyncio.run(start_ingestion())

if __name__ == "__main__":
    main()
