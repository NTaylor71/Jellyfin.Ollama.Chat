#!/usr/bin/env python3
"""
Minimal test to isolate the 307 issue
"""
import asyncio
import httpx
from rich.console import Console

console = Console()


async def test_chat_endpoint():
    """Test the chat endpoint directly"""
    console.print("🧪 Minimal Chat Endpoint Test", style="bold blue")

    url = "http://localhost:8000/chat/"
    payload = {"query": "minimal test"}

    console.print(f"📤 URL: {url}", style="cyan")
    console.print(f"📋 Payload: {payload}", style="cyan")

    try:
        async with httpx.AsyncClient() as client:
            console.print("🔗 Making request...", style="cyan")

            response = await client.post(
                url,
                json=payload,
                timeout=10.0
            )

            console.print(f"📨 Status: {response.status_code}", style="green")
            console.print(f"📨 Headers: {dict(response.headers)}", style="cyan")
            console.print(f"📨 Text: {response.text}", style="cyan")

            if response.status_code == 307:
                console.print(f"🔄 Redirect to: {response.headers.get('location')}", style="yellow")

            return response.status_code == 200

    except Exception as e:
        console.print(f"💥 Error: {e}", style="red")
        return False


async def main():
    result = await test_chat_endpoint()
    console.print(f"\n✅ Test {'PASSED' if result else 'FAILED'}", style="bold green" if result else "bold red")


if __name__ == "__main__":
    asyncio.run(main())
