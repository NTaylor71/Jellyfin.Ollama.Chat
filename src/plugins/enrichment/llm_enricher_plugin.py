import re
import json
from ollama import AsyncClient
from colorama import Fore, Style

from src.config import OLLAMA_CHAT_MODEL, OLLAMA_CHAT_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_EMBED_BASE_URL
from src.plugins.embedding.cpu_embedder_plugin import CPUEmbedderPlugin


class LLMEnricherPlugin:
    def __init__(self):
        self.client = AsyncClient(host=OLLAMA_CHAT_BASE_URL)
        self.model = OLLAMA_CHAT_MODEL
        print(f"Movie Enricher: {self.model} @ {OLLAMA_CHAT_BASE_URL}")

    def _extract_json(self, raw: str) -> str:
        """Try to extract the first JSON-looking object from messy LLM output."""
        match = re.search(r'{.*}', raw, re.DOTALL)
        return match.group(0).strip() if match else raw.strip()

    def _validate_fields(self, parsed: dict) -> bool:
        required_keys = [
            "summary", "censorship", "genre_commentary",
            "viewer_summaries", "reviews", "historical_context"
        ]
        if not all(k in parsed for k in required_keys):
            return False

        if not isinstance(parsed["viewer_summaries"], list) or not isinstance(parsed["reviews"], list):
            return False

        return all(isinstance(parsed[k], str) for k in parsed if k not in ("viewer_summaries", "reviews"))

    def build_full_text(self, fields: dict) -> str:
        def section(title, content, icon):
            if not content:
                return ""
            if isinstance(content, list):
                bullets = "\n".join(f"- {line.strip()}" for line in content if line.strip())
                return f"{icon} {title}:\n{bullets}" if bullets else ""
            return f"{icon} {title}:\n{content.strip()}"

        return "\n\n".join(filter(None, [
            section("Summary", fields.get("summary"), "📝"),
            section("Censorship", fields.get("censorship"), "🙊"),
            section("Genre Commentary", fields.get("genre_commentary"), "🔍"),
            section("Viewer Summaries", fields.get("viewer_summaries"), "🎭"),
            section("Reviews", fields.get("reviews"), "📝"),
            section("Historical Context", fields.get("historical_context"), "📚")
        ])).strip()

    async def enrich_movie(self, rendered_text: str, title: str = "unknown") -> dict:
        base_prompt = f"""
You are a metadata enhancer for a movie search engine.

Given this movie block:
{rendered_text}

Return VALID JSON ONLY with all 6 of the following fields:
{{
  "summary": "...",
  "censorship": "...",
  "genre_commentary": "...",
  "viewer_summaries": ["...", "..."],
  "reviews": ["...", "..."],
  "historical_context": "..."
}}

STRICT RULES:
- DO NOT use markdown, quotes, or code blocks
- DO NOT explain — return JSON only
- All fields MUST be present and valid
"""

        for attempt in range(1, 11):
            try:
                temperature = 0.3 if attempt > 2 else 0.5
                prompt = base_prompt
                if attempt > 5:
                    prompt = (
                        "Your previous reply was invalid. Please return only JSON.\n\n" + base_prompt
                    )

                response = await self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature},
                )
                raw = response["message"]["content"].strip()
                clean = self._extract_json(raw)

                parsed = json.loads(clean)
                if not self._validate_fields(parsed):
                    raise ValueError("Invalid field types or missing keys")

                # Normalize lists
                for k in ("viewer_summaries", "reviews"):
                    parsed[k] = [x.strip() for x in parsed.get(k, []) if isinstance(x, str) and x.strip()]

                for k in parsed:
                    if isinstance(parsed[k], str):
                        parsed[k] = parsed[k].strip()

                # Embed each enriched field
                embedder = CPUEmbedderPlugin(model=OLLAMA_EMBED_MODEL, host=OLLAMA_EMBED_BASE_URL)
                vector_fields = {}
                for k in ["summary", "genre_commentary", "viewer_summaries", "reviews", "historical_context"]:
                    content = parsed.get(k)
                    if not content:
                        continue
                    text = "\n".join(content) if isinstance(content, list) else content
                    vec = (await embedder.embed_texts([text]))[0]
                    vector_fields[k] = vec

                return {
                    "fields": parsed,
                    "full_text": self.build_full_text(parsed),
                    "vector_fields": vector_fields
                }

            except Exception as e:
                with open("enrich_failed.log", "a", encoding="utf-8") as f:
                    f.write(f"\n=== FAILED [{title}] Attempt {attempt} ===\n")
                    f.write(f"Prompt:\n{prompt.strip()}\n\n")
                    f.write(f"Raw LLM Output:\n{raw.strip() if 'raw' in locals() else '[no output]'}\n")
                    f.write(f"Error: {str(e)}\n")

                print(f"{Fore.RED}⚠️ Attempt {attempt}/10 failed for '{title}': {e}{Style.RESET_ALL}")

        print(f"{Fore.RED}❌ Giving up on '{title}' after 10 failures.{Style.RESET_ALL}")
        return {
            "fields": {
                "summary": "",
                "censorship": "",
                "genre_commentary": "",
                "viewer_summaries": [],
                "reviews": [],
                "historical_context": ""
            },
            "full_text": rendered_text.strip(),
            "vector_fields": {}
        }
