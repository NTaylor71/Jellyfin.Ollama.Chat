import asyncio
import argparse
import json
import re
import pickle
from pathlib import Path
import numpy as np
from colorama import Fore, Style, init as colorama_init
import faiss

from src.plugins.enrichment.llm_enricher_plugin import LLMEnricherPlugin
from src.plugins.embedding.cpu_embedder_plugin import CPUEmbedderPlugin
from src.config import OLLAMA_EMBED_MODEL, OLLAMA_EMBED_BASE_URL

CACHE_FILE = Path("embedding_cache.pkl")
FAISS_FILE = Path("faiss.index")
BACKUP_CACHE = Path("embedding_cache.bak.pkl")

colorama_init(autoreset=True)


def extract_failures_from_log(log_path: Path) -> list[dict]:
    blocks = log_path.read_text(encoding="utf-8").split("=== FAILED ")
    extracted = []

    for block in blocks:
        if not block.strip():
            continue
        title_match = re.match(r"\[(.*?)\]", block)
        title = title_match.group(1).strip() if title_match else "unknown"
        prompt_match = re.search(r"Prompt:\n(.*?)\n\nRaw LLM Output:", block, re.DOTALL)
        prompt = prompt_match.group(1).strip() if prompt_match else None
        if title and prompt:
            extracted.append({"title": title, "prompt": prompt})
    return extracted


def load_existing_cache() -> list[dict]:
    if not CACHE_FILE.exists():
        return []
    with CACHE_FILE.open("rb") as f:
        return pickle.load(f)


def save_updated_cache(cache: list[dict]):
    if CACHE_FILE.exists():
        CACHE_FILE.replace(BACKUP_CACHE)
        print(f"{Fore.YELLOW}📦 Backup saved: {BACKUP_CACHE}")
    with CACHE_FILE.open("wb") as f:
        pickle.dump(cache, f)
    print(f"{Fore.GREEN}✅ Updated cache written to: {CACHE_FILE}")


def rebuild_faiss_index(entries: list[dict]):
    print(f"{Fore.CYAN}🔧 Rebuilding FAISS index...")
    vectors = np.array([entry["vector"] for entry in entries], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(FAISS_FILE))
    print(f"{Fore.GREEN}✅ FAISS index rebuilt and saved.")


async def retry_failures(failures: list[dict], existing: list[dict], retry: bool = False):
    enricher = LLMEnricherPlugin()
    embedder = CPUEmbedderPlugin(model=OLLAMA_EMBED_MODEL, host=OLLAMA_EMBED_BASE_URL)
    existing_titles = {e.get("title") for e in existing}
    new_entries = []

    for i, failure in enumerate(failures):
        title = failure["title"]
        prompt = failure["prompt"]
        if title in existing_titles:
            print(f"{Fore.CYAN}⏭️ Skipping duplicate: {title}")
            continue

        print(f"\n🔁 Retrying [{i + 1}/{len(failures)}]: {title}")
        try:
            enriched = await enricher.enrich_movie(rendered_text="[REPAIR]", title=title)
            if enriched and enriched["fields"]["summary"]:
                print(f"{Fore.GREEN}✅ Success: {title}")
                vec = (await embedder.embed_texts([enriched["full_text"]]))[0]
                enriched.update({
                    "title": title,
                    "real_meta": "[Recovered from enrich_failed.log]",
                    "vector": vec
                })
                new_entries.append(enriched)
            else:
                print(f"{Fore.YELLOW}⚠️ Skipped: No usable summary for {title}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error retrying {title}: {e}{Style.RESET_ALL}")

    if retry and new_entries:
        print(f"\n{Fore.YELLOW}📝 Merging {len(new_entries)} repaired entries into cache...")
        updated = existing + new_entries
        save_updated_cache(updated)
        rebuild_faiss_index(updated)
    else:
        print(f"\n{Fore.CYAN}Dry run complete. Use --retry to update cache.")

    return new_entries


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="enrich_failed.log", help="Log file to process")
    parser.add_argument("--retry", action="store_true", help="Actually update cache/index")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"{Fore.RED}❌ Log file not found: {log_path}")
        return

    failures = extract_failures_from_log(log_path)
    if not failures:
        print(f"{Fore.GREEN}✅ No retryable entries found.")
        return

    print(f"{Fore.MAGENTA}🔍 Found {len(failures)} failures to retry.")
    existing = load_existing_cache()
    await retry_failures(failures, existing, retry=args.retry)


if __name__ == "__main__":
    asyncio.run(main())
