import json
from ollama import AsyncClient
from src.config import ENRICH_CHAT_MODEL, ENRICH_CHAT_BASE_URL
from colorama import Fore, Style


class LLMPromptEnricherPlugin:
    """
    Enhances a user's query for movie search by returning:
    - A rewritten paragraph version (for embedding)
    - A list of topic cluster keywords (for boosting/filtering)
    """

    def __init__(self):
        self.client = AsyncClient(host=ENRICH_CHAT_BASE_URL)
        self.model = ENRICH_CHAT_MODEL
        print(f"Prompt Enricher: {self.model} @ {ENRICH_CHAT_BASE_URL}")

    async def enrich_query(self, query: str, debug: bool = False) -> dict:
        """
        Expands a query using LLM into a paragraph + keyword list.
        Retries up to 3x if invalid format returned.
        """

        base_prompt = f"""
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

        for attempt in range(1, 4):
            try:
                temperature = 0.5 if attempt == 1 else 0.3
                prompt = base_prompt if attempt == 1 else (
                        "⚠️ You previously failed to return valid JSON.\n"
                        "Try again, ONLY return JSON. Do not explain.\n\n" + base_prompt
                )

                response = await self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature},
                )
                raw = response["message"]["content"].strip()
                print(f"{Fore.LIGHTBLACK_EX}📝 Raw LLM response (attempt {attempt}):\n{raw[:300]}{Style.RESET_ALL}")

                if raw.startswith("<think>"):
                    raw = re.sub(r"(?s)^<think>.*?\n", "", raw).strip()
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
                print(f"{Fore.RED}⚠️ Attempt {attempt} failed: {e}{Style.RESET_ALL}")
                continue

        # Final fallback if all attempts fail
        fallback_keywords = [w.strip(",.") for w in query.lower().split() if len(w) > 4]
        return {
            "query_paragraph": query,
            "keywords": fallback_keywords
        }
