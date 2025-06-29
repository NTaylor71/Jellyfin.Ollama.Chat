import asyncio
import argparse
import json
import pickle
import re
import time
from pathlib import Path
from typing import List
import numpy as np
from ollama import AsyncClient
from colorama import Fore, Style, init as colorama_init
from src.data.sample_entries import get_sample_entries
from src.rag.formatter import render_media_text_block
from src.plugins.enrichment.llm_enricher_plugin import LLMEnricherPlugin
from src.plugins.enrichment.llm_prompt_enricher_plugin import LLMPromptEnricherPlugin
from src.plugins.embedding.cpu_embedder_plugin import CPUEmbedderPlugin
import faiss

from src.config import (
    OLLAMA_EMBED_MODEL,
    OLLAMA_EMBED_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_CHAT_BASE_URL,
)

import nltk
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

colorama_init(autoreset=True)

ENRICHED_TEXT_FIELDS = [
    "summary",
    "genre_commentary",
    "viewer_summaries",
    "reviews",
    "historical_context",
]

CACHE_FILE = Path("embedding_cache.pkl")
FAISS_INDEX_FILE = Path("faiss.index")

DEFAULT_TOP_N = 3
DEFAULT_KEYWORD_WEIGHT = 0.05
HYBRID_ALPHA = 0.7

def top_keyword_field_matches(
    entry: dict,
    keyword_vectors: dict[str, list[float]],
    threshold: float = 0.65
) -> list[tuple[str, str, float]]:
    """Returns list of (keyword, field, similarity) above threshold."""
    matches = []
    field_vectors = entry.get("vector_fields", {})
    for field, field_vec in field_vectors.items():
        field_np = np.array(field_vec, dtype=np.float32)
        field_np /= np.linalg.norm(field_np)
        for keyword, kw_vec in keyword_vectors.items():
            kw_np = np.array(kw_vec, dtype=np.float32)
            kw_np /= np.linalg.norm(kw_np)
            similarity = float(np.dot(field_np, kw_np))
            if similarity >= threshold:
                matches.append((keyword, field, similarity))
    return matches


def normalize_keyword(kw: str) -> str:
    text = re.sub(r"[^\w\s]", "", kw.lower())
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())


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


def load_entries(input_file: str | None) -> list[dict]:
    print(f"{Fore.CYAN}📦 Loading sample entries...")
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)
        print(f"{Fore.GREEN}✅ Loaded entries from file: {input_file}")
    else:
        raw_entries = get_sample_entries()
        print(f"{Fore.GREEN}✅ Loaded built-in sample entries")
    return raw_entries


def render_entries(raw_entries: list[dict]) -> list[str]:
    print(f"\n{Fore.CYAN}🧠 Rendering raw entries using Jinja2...")
    return [render_media_text_block(entry) for entry in raw_entries]


async def enrich_and_embed_entries(rendered: list[str], raw_entries: list[dict], debug_enrich: bool = False) -> list[dict]:
    print(f"\n{Fore.CYAN}🔁 Enriching and embedding sequentially...")
    enricher = LLMEnricherPlugin()
    embedder = CPUEmbedderPlugin(model=OLLAMA_EMBED_MODEL, host=OLLAMA_EMBED_BASE_URL)

    enriched_entries = []
    start = time.time()

    for i, rendered_text in enumerate(rendered):
        title = raw_entries[i].get("title", f"Movie {i+1}")
        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        remaining = len(rendered) - (i + 1)
        eta = format_duration(avg * remaining)

        print(f"{Fore.YELLOW}🧪 [{i+1}/{len(rendered)}] {title} — ETA: {eta}, avg: {avg:.2f}s", end="", flush=True)
        t0 = time.time()

        try:
            enriched = await enricher.enrich_movie(rendered_text, title=title)
            enriched["title"] = title
            enriched["year"] = raw_entries[i].get("year")
            enriched["real_meta"] = rendered_text
            vec = (await embedder.embed_texts([enriched["full_text"]]))[0]
            enriched["vector"] = vec
            print(f"{Fore.GREEN} — Done ({format_duration(time.time() - t0)})")

            if debug_enrich:
                print(f"{Fore.CYAN}🔍 Enriched Result:\n{Style.RESET_ALL}{enriched['full_text']}\n")

            enriched_entries.append(enriched)

        except Exception as e:
            print(f"{Fore.RED}⚠️ Failed for '{title}': {e}")

    print(f"{Fore.GREEN}✅ All entries processed: {len(enriched_entries)} successful [{format_duration(time.time() - start)}].")
    return enriched_entries


def save_cache(entries: List[dict]):
    with CACHE_FILE.open("wb") as f:
        pickle.dump(entries, f)
    print(f"{Fore.GREEN}📀 Saved enriched entries to {CACHE_FILE}")


def load_cache() -> List[dict]:
    with CACHE_FILE.open("rb") as f:
        return pickle.load(f)


def build_faiss_index(entries: List[dict]) -> faiss.IndexFlatIP:
    print(f"{Fore.CYAN}🔧 Building FAISS index...")
    vectors = np.array([entry["vector"] for entry in entries]).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    print(f"{Fore.GREEN}✅ FAISS index saved to {FAISS_INDEX_FILE}")
    return index


def load_faiss_index(entries: List[dict]) -> faiss.IndexFlatIP:
    if FAISS_INDEX_FILE.exists():
        print(f"{Fore.CYAN}📂 Loading FAISS index from disk...")
        return faiss.read_index(str(FAISS_INDEX_FILE))
    return build_faiss_index(entries)


def matched_keywords_by_field_vector(
        entry: dict,
        keyword_vectors: dict[str, list[float]],
        similarity_threshold: float = 0.65) -> dict[str, list[str]]:
    results = {}
    field_vectors = entry.get("vector_fields", {})

    for field, field_vec in field_vectors.items():
        field_np = np.array(field_vec, dtype=np.float32)
        field_np /= np.linalg.norm(field_np)

        matched = []
        for keyword, kw_vec in keyword_vectors.items():
            kw_np = np.array(kw_vec, dtype=np.float32)
            kw_np /= np.linalg.norm(kw_np)
            similarity = float(np.dot(field_np, kw_np))
            print(f"    ↪️ {keyword} vs {field} → {similarity:.3f}")
            if similarity >= similarity_threshold:
                print(f"    ↪️ Met threshold")
                matched.append(keyword)

        if matched:
            results[field] = matched

    return results


async def expand_keywords_with_llm(keywords: list[str]) -> dict[str, list[str]]:
    client = AsyncClient(host=OLLAMA_CHAT_BASE_URL)
    prompt = f"""
Given the list of keywords below, return expanded terms or close synonyms.

Respond ONLY with JSON like:
{{ "expanded": {{ "keyword1": ["syn1", "syn2"], ... }} }}

List:
{json.dumps(keywords)}
"""
    try:
        response = await client.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.5},
        )
        raw = response["message"]["content"].strip("`\n ")
        return json.loads(raw).get("expanded", {})
    except Exception as e:
        print(f"{Fore.RED}⚠️ Keyword synonym expansion failed: {e}{Style.RESET_ALL}")
        return {}


async def enrich_prompt(query: str, mode: str, debug: bool) -> tuple[str, list[str], list[float], dict[str, list[float]]]:
    if mode == "literal":
        vec = await embed_query(query)
        return query, [], vec, {}

    enricher = LLMPromptEnricherPlugin()
    result = await enricher.enrich_query(query, debug=debug)
    paragraph = result["query_paragraph"]
    base_keywords = result["keywords"]

    synonym_map = await expand_keywords_with_llm(base_keywords)
    enriched_keywords = list(set(base_keywords + [syn for syns in synonym_map.values() for syn in syns]))

    if debug:
        print(f"{Fore.CYAN}🧠 Keyword Expansion:\n{json.dumps(synonym_map, indent=2)}{Style.RESET_ALL}")

    embedder = CPUEmbedderPlugin(model=OLLAMA_EMBED_MODEL, host=OLLAMA_EMBED_BASE_URL)

    # ✅ Normalize keywords before embedding
    normalized_keywords = [normalize_keyword(kw) for kw in enriched_keywords]
    keyword_vecs = await embedder.embed_texts(normalized_keywords)
    keyword_vectors = dict(zip(enriched_keywords, keyword_vecs))  # preserve original names for display

    vec_verbose = (await embedder.embed_texts([paragraph]))[0]

    if mode == "hybrid":
        vec_literal = (await embedder.embed_texts([query]))[0]
        vec = (HYBRID_ALPHA * np.array(vec_verbose) + (1 - HYBRID_ALPHA) * np.array(vec_literal)).tolist()
    else:
        vec = vec_verbose

    return paragraph, enriched_keywords, vec, keyword_vectors


async def embed_query(text: str) -> list[float]:
    embedder = CPUEmbedderPlugin(model=OLLAMA_EMBED_MODEL, host=OLLAMA_EMBED_BASE_URL)
    return (await embedder.embed_texts([text]))[0]


def show_matches(entries: List[dict], index: faiss.IndexFlatIP, query_vec: list[float], keywords: list[str], keyword_vectors: dict[str, list[float]], top_n: int, keyword_weight: float, show_enriched: bool, debug_json: bool):
    query_np = np.array([query_vec]).astype("float32")
    query_np /= np.linalg.norm(query_np)
    D, I = index.search(query_np, top_n * 2)

    results = []
    for idx, base_score in zip(I[0], D[0]):
        if idx >= len(entries):
            continue
        entry = entries[idx]
        field_matches = matched_keywords_by_field_vector(entry, keyword_vectors)
        top_matches = top_keyword_field_matches(entry, keyword_vectors)
        boost = sum((sim - 0.65) for _, _, sim in top_matches) * keyword_weight

        results.append((idx, base_score + boost, field_matches))

    top = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n{Fore.MAGENTA}🔝 Top {top_n} Matches:")
    for rank, (idx, score, matched) in enumerate(top, 1):
        entry = entries[idx]
        title = entry.get("title", f"Movie {idx}")
        year = entry.get("year")
        header = f"{title} ({year})" if year else title

        print(f"{Fore.YELLOW}#{rank} 🎬 {header} | Score: {score:.4f}")
        print(f"{Fore.GREEN}🌟 Real Metadata:\n{Style.RESET_ALL}{entry.get('real_meta', '').strip()}\n")

        print(f"{Fore.GREEN}📚 Enriched Text Fields:{Style.RESET_ALL}")
        fields = entry.get("fields", {})

        def show_field(label, key, icon, is_list=False):
            content = fields.get(key)
            if is_list:
                if content:
                    print(f"{icon} {label}:")
                    for line in content:
                        print(f"- {line}")
                else:
                    print(f"{icon} {label}: None provided")
            else:
                print(f"{icon} {label}: {content.strip() if content else 'None provided'}")

        show_field("Summary", "summary", "📝")
        show_field("Censorship", "censorship", "🚊")
        show_field("Genre Commentary", "genre_commentary", "🔍")
        show_field("Viewer Summaries", "viewer_summaries", "🎭", is_list=True)
        show_field("Reviews", "reviews", "📝", is_list=True)
        show_field("Historical Context", "historical_context", "📚")

        if show_enriched and "enriched_text" in entry:
            print(f"\n{Fore.CYAN}🔮 Enriched Extra:\n{entry['enriched_text']}\n")

        print(f"{Fore.BLUE}💬 Matched keywords:{Style.RESET_ALL}")
        if debug_json:
            top_matches_sorted = sorted(top_matches, key=lambda x: x[2], reverse=True)[:10]
            print(f"{Fore.LIGHTBLACK_EX}🔢 Top Keyword↔️Field Matches:")

            for keyword, field, sim in top_matches_sorted:
                if sim >= 0.75:
                    color = Fore.GREEN
                elif sim >= 0.65:
                    color = Fore.YELLOW
                else:
                    color = Fore.LIGHTBLACK_EX
                print(f"{color} - {keyword} → {field} : {sim:.3f}")

            print(Style.RESET_ALL)

        if matched:
            for field, kws in matched.items():
                print(f" - {field}: {', '.join(kws)}")
        else:
            print(" - None found.")

        if debug_json:
            matched_keywords = set(kw for kws in matched.values() for kw in kws)
            missed_keywords = [kw for kw in keywords if kw not in matched_keywords]
            print(f"{Fore.LIGHTBLACK_EX}📎 Field JSON:\n{json.dumps(fields, indent=2)}{Style.RESET_ALL}")
            if missed_keywords:
                print(f"{Fore.LIGHTBLACK_EX}🗣️ Unmatched Keywords: {missed_keywords}{Style.RESET_ALL}")

        print("")


async def main(args):
    t0 = time.time()

    if args.reset_cache:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            print(f"{Fore.YELLOW}🪩 Deleted {CACHE_FILE}")
        if FAISS_INDEX_FILE.exists():
            FAISS_INDEX_FILE.unlink()
            print(f"{Fore.YELLOW}🪩 Deleted {FAISS_INDEX_FILE}")
        if not args.query:
            print(f"{Fore.GREEN}✅ Cache cleared. No query provided — exiting.")
            return

    if not args.query:
        print(f"{Fore.RED}❌ Error: --query is required unless using --reset-cache alone.")
        return

    if args.skip_embed and CACHE_FILE.exists() and FAISS_INDEX_FILE.exists():
        entries = load_cache()
        index = load_faiss_index(entries)
    else:
        raw = load_entries(args.input_file)
        rendered = render_entries(raw)

        if args.max_embeds:
            rendered = rendered[:args.max_embeds]
            raw = raw[:args.max_embeds]
            print(f"{Fore.CYAN}🔁 Trimming to first {args.max_embeds} entries...")

        entries = await enrich_and_embed_entries(rendered, raw, debug_enrich=args.debug_enrich)
        save_cache(entries)
        index = build_faiss_index(entries)

    paragraph, keywords, query_vec, keyword_vectors = await enrich_prompt(args.query, args.query_mode, args.debug_query)
    print(f"\n{Fore.CYAN}🔍 Searching for query: '{paragraph}'...")
    if args.debug_json:
        print(f"{Fore.LIGHTBLACK_EX}🔍 Matching fields used:\n{ENRICHED_TEXT_FIELDS}{Style.RESET_ALL}")

    show_matches(entries, index, query_vec, keywords, keyword_vectors, args.top_n, args.keyword_weight, args.show_enriched, args.debug_json)
    print(f"{Fore.GREEN}✅ Search complete")
    print(f"{Fore.CYAN}⏱️ Time: {format_duration(time.time() - t0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-embed", action="store_true", help="Use cached embeddings if available")
    parser.add_argument("--reset-cache", action="store_true", help="Delete prior cache and index")
    parser.add_argument("--max-embeds", type=int, help="Limit how many movie entries to process")
    parser.add_argument("--query", type=str, help="Search query (required unless using --reset-cache only)")
    parser.add_argument("--input-file", type=str, help="Path to movie JSON input")
    parser.add_argument("--show-enriched", action="store_true", help="Show enriched LLM blob (if available)")
    parser.add_argument("--query-mode", choices=["literal", "verbose", "hybrid"], default="verbose")
    parser.add_argument("--debug-query", action="store_true", help="Show enriched prompt + keywords")
    parser.add_argument("--debug-enrich", action="store_true", help="Show enriched movie full_text blocks during processing")
    parser.add_argument("--debug-json", action="store_true", help="Print full enriched JSON")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    args = parser.parse_args()

    asyncio.run(main(args))
