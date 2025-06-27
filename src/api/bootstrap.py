import asyncio
import signal
import sys
from .safe_restart import safe_restart
from .plugin_loader import load_plugins

def bootstrap():
    print("🔧 Bootstrapping FAISS RAG API...")

    # Initial plugin load
    load_plugins()

    # Register signal handlers for safe restart
    def handle_signal(sig, frame):
        print(f"⚙️ Caught signal: {sig}. Triggering safe restart...")
        asyncio.run(safe_restart())

    signal.signal(signal.SIGHUP, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("✅ Bootstrap complete. API is ready.")

if __name__ == "__main__":
    bootstrap()
