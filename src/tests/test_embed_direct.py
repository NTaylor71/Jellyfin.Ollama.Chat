import asyncio
import argparse
import json
import numpy as np
from pprint import pprint
from typing import List
import pickle
from pathlib import Path
import time

from colorama import Fore, Style, init as colorama_init
from src.data.sample_entries import get_sample_entries
from src.rag.formatter import render_media_text_block
from src.plugins.enrichment.llm_enricher_plugin import LLMEnricherPlugin
from src.plugins.enrichment.llm_prompt_enricher_plugin import LLMPromptEnricherPlugin
from src.plugins.embedding.cpu_embedder_plugin import CPUEmbedderPlugin
from ollama import AsyncClient

colorama_init(autoreset=True)
EMBED_CACHE_PATH = Path("embedding_cache.pkl")


def cosine_similarity(a: List[float], b: List[List[float]]) -> List[float]:
    a_np = np.array(a)
    b_np = np.array(b)
    a_norm = a_np / np.linalg.norm(a_np)
    b_norm = b_np / np.linalg.norm(b_np, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, min = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {min}m"
    days, hr = divmod(hours, 24)
    return f"{days}d {hr}h"


async def main(skip_embed: bool, query: str, input_file: str | None):
    t0 = time.time()
    if not skip_embed:
        print(f"{Fore.CYAN}📦 Loading sample entries...")

        if input_file:
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    raw_entries = json.load(f)
                print(f"{Fore.GREEN}✅ Loaded entries from file: {input_file}")
            except Exception as e:
                print(f"{Fore.RED}❌ Failed to read input file: {e}")
                return
        else:
            raw_entries = get_sample_entries()
            print(f"{Fore.GREEN}✅ Loaded built-in sample entries")

        print(f"{Fore.GREEN}✅ Raw Entries ({len(raw_entries)} loaded) [{format_duration(time.time() - t0)}]")
        pprint(raw_entries)

    if not skip_embed:
        t1 = time.time()
        print(f"\n{Fore.CYAN}🧠 Rendering raw entries into natural language using Jinja2...")
        rendered = [render_media_text_block(entry) for entry in raw_entries]
        print(f"{Fore.GREEN}✅ Rendered Entries [{format_duration(time.time() - t1)}]")
        for i, r in enumerate(rendered):
            print(f"\n{Fore.YELLOW}🔹 Entry {i + 1}:\n{Style.RESET_ALL}{r}")

    if skip_embed and EMBED_CACHE_PATH.exists():
        t2 = time.time()
        print(f"\n{Fore.CYAN}⏭️ Skipping enrichment/embedding and loading from cache...")
        with EMBED_CACHE_PATH.open("rb") as f:
            enriched_texts, vectors = pickle.load(f)
        print(f"{Fore.GREEN}✅ Loaded from cache [{format_duration(time.time() - t2)}]")
    else:
        t2 = time.time()
        print(f"\n{Fore.CYAN}🤖 Enriching rendered text via LLM (GPU)...")
        enricher = LLMEnricherPlugin()
        enriched_texts = []

        start = time.time()
        for i, text in enumerate(rendered):
            remaining = len(rendered) - i
            elapsed = time.time() - start
            avg_time = elapsed / (i or 1)
            eta = format_duration(avg_time * remaining)

            if i:
                print(f"{Fore.YELLOW}🧪 [{i + 1}/{len(rendered)}] Enriching entry... ETA: {eta}")
            else:
                print(f"{Fore.YELLOW}🧪 [{i + 1}/{len(rendered)}] Enriching entry...")

            try:
                entry_start = time.time()
                enriched = await enricher.enrich(text)
                duration = format_duration(time.time() - entry_start)
                print(f"{Fore.GREEN}✅ [{i + 1}] Done in {duration}")
            except Exception as e:
                print(f"{Fore.RED}❌ [{i + 1}] Enrichment failed: {e}")
                enriched = text  # fallback

            enriched_texts.append(enriched)

        print(f"{Fore.GREEN}✅ All entries enriched [{format_duration(time.time() - t2)}]")

        t3 = time.time()
        print(f"\n{Fore.CYAN}📨 Sending enriched texts to embedding plugin (CPU)...")
        embedder = CPUEmbedderPlugin()
        vectors = await embedder.embed_texts(enriched_texts)

        if not (isinstance(vectors, list) and all(isinstance(v, list) for v in vectors)):
            print(f"{Fore.RED}❌ Embedding plugin returned invalid output.")
            print(vectors)
            return

        for i, vector in enumerate(vectors):
            print(f"{Fore.GREEN}✅ Embedding {i + 1} | Length: {len(vector)}")
            print(f"{Fore.YELLOW}🔹 Preview: {vector[:5]} ...")

        print(f"{Fore.GREEN}✅ All embeddings complete [{format_duration(time.time() - t3)}]")

        with EMBED_CACHE_PATH.open("wb") as f:
            pickle.dump((enriched_texts, vectors), f)
        print(f"{Fore.GREEN}💾 Cached enriched texts and embeddings to disk.")


    # -- Prompt Enricher
    t2 = time.time()
    print(f"\n{Fore.CYAN}🤖 Enriching prompt text via LLM (GPU)...")
    prompt_enricher = LLMPromptEnricherPlugin()

    try:
        entry_start = time.time()
        enriched_prompt = await prompt_enricher.enrich(query)
        duration = format_duration(time.time() - entry_start)
        print(f"{Fore.GREEN}✅ Done in {duration}")
    except Exception as e:
        print(f"{Fore.RED}❌ Enrichment failed: {e}")
        enriched_prompt = query  # fallback

    t4 = time.time()
    print(f"\n{Fore.CYAN}🔍 Searching for query: '{enriched_prompt}'...")
    query_client = AsyncClient(host="http://localhost:12435")
    query_result = await query_client.embed(model="nomic-embed-text", input=[query])
    query_vec = query_result["embeddings"][0]

    scores = cosine_similarity(query_vec, vectors)

    # Get top 3 indices
    top_indices = np.argsort(scores)[-3:][::-1]  # Sorted descending

    # Print top 3 matches
    print(f"\n{Fore.MAGENTA}🔝 Top 3 Matches:")
    for rank, idx in enumerate(top_indices, start=1):
        print(f"{Fore.YELLOW}#{rank} Index: {idx} | Score: {scores[idx]:.4f}")
        print(f"{Fore.GREEN}🎯 Match:\n{Style.RESET_ALL}{enriched_texts[idx].split('---')[0]}\n")

    print(f"{Fore.GREEN}✅ Search complete [{format_duration(time.time() - t4)}]")
    print(f"{Fore.CYAN}⏱️ Total time: {format_duration(time.time() - t0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-embed", action="store_true", help="Use cached embeddings")
    parser.add_argument("--query", type=str, default="mind bending scifi movies", help="Search query string")
    parser.add_argument("--input-file", type=str, help="Optional path to JSON file with raw entries")
    args = parser.parse_args()

    asyncio.run(main(skip_embed=args.skip_embed, query=args.query, input_file=args.input_file))
