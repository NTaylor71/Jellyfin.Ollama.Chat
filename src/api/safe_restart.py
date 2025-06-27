import asyncio
import os
import signal
import sys

from .plugin_loader import load_plugins

async def safe_restart():
    """Reload plugins without restarting the entire container."""
    try:
        print("🔁 Performing safe restart: Reloading plugins...")
        load_plugins()
        print("✅ Plugins reloaded successfully.")
    except Exception as e:
        print(f"❌ Plugin reload failed: {e}")
        print("🔄 Restarting API process to recover...")
        await force_restart()

async def force_restart():
    """Force restart the current process cleanly."""
    await asyncio.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM)
    # If SIGTERM fails, exit immediately (rare but safe fallback)
    sys.exit(1)
