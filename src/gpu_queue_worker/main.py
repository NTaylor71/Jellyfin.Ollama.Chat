# /src/gpu_queue_worker/main.py

import asyncio
from src.gpu_queue_worker.gpu_worker import gpu_worker_loop

def main():
    print("🚀 GPU Queue Worker starting loop...")
    asyncio.run(gpu_worker_loop())

if __name__ == "__main__":
    main()
