import json
from ollama import AsyncClient
from src.config import OLLAMA_CHAT_MODEL, OLLAMA_CHAT_BASE_URL
from colorama import Fore, Style


class LLMPromptEnricherPlugin:
    """
    Enhances a user's query for movie search by returning:
    - A rewritten paragraph version (for embedding)
    - A list of topic cluster keywords (for boosting/filtering)
    """

    def __init__(self):
        self.client = AsyncClient(host=OLLAMA_CHAT_BASE_URL)
        self.model = OLLAMA_CHAT_MODEL
        print(f"Prompt Enricher: {self.model} @ {OLLAMA_CHAT_BASE_URL}")

    async def enrich_query(self, query: str, debug: bool = False) -> dict:
        """
        Expands a query using LLM into a paragraph + keyword list.
        """
        prompt = f"""
You are a prompt enhancer for a movie semantic search engine.

Given the user's vague or thematic query, return:
- A rewritten paragraph that expands the intent
- A list of keywords related to the topic

Respond ONLY with JSON:
{{
  "query_paragraph": "Expanded natural language paragraph.",
  "keywords": ["term1", "term2", "term3"]
}}

⚠️ STRICT RULES:
- DO NOT include markdown (no backticks or code blocks)
- DO NOT explain or comment — only return JSON
- DO NOT omit fields — both fields must be present

User Query:
{query}
"""

        try:
            response = await self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.5},
            )
            raw = response["message"]["content"].strip()

            if raw.startswith("```") or raw.startswith("{"):
                raw = raw.strip("` \n")

            parsed = json.loads(raw)

            if not parsed.get("query_paragraph") or not isinstance(parsed.get("keywords"), list):
                raise ValueError("Missing expected fields in JSON")

            if debug:
                print(f"\n{Fore.YELLOW}🔎 Enriched Query:{Style.RESET_ALL}\n{parsed['query_paragraph']}")
                print(f"{Fore.YELLOW}🔑 Keywords:{Style.RESET_ALL} {', '.join(parsed['keywords'])}\n")

            return parsed

        except Exception as e:
            print(f"{Fore.RED}⚠️ Failed to parse LLM prompt enrichment: {e}{Style.RESET_ALL}")
            return {
                "query_paragraph": query,
                "keywords": []
            }
