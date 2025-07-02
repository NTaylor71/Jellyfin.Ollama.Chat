# synonyms/base.py  – v3.5 (full file)
"""
Robust, extensible synonym generator.

New in v3.5
───────────
* **Output hygiene**
  • Drops single‑character tokens (e.g. “d”)
  • Drops common stop‑words (“and/the/is…”)
  • Ensures terms are unique per method after filtering.
* **LLM weighting**
  • Assigns descending weights (1.0,0.9,0.8…) to the list returned by Ollama.
  • If a gensim embedding model is loaded, overrides that heuristic with the
    actual cosine similarity (`token↔term`) just like the embeddings method.
* **Embeddings duplicates removed** (console/console second “console”).
All earlier features—timings, cache, WordNet frequency weights—are retained.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from ollama import AsyncClient

    _LLM_AVAILABLE = True
except ImportError as _e:
    AsyncClient = None  # type: ignore
    _LLM_AVAILABLE = False
    logging.getLogger("synonyms").warning(
        "LLM disabled – ollama not importable (%s). "
        "Install `ollama` Python client if you want LLM synonyms.",
        _e,
    )

try:
    from gensim.models import KeyedVectors
    import gensim.downloader as api

    _EMBEDDINGS_AVAILABLE = True
except ImportError as _e:
    KeyedVectors = None  # type: ignore
    api = None
    _EMBEDDINGS_AVAILABLE = False
    logging.getLogger("synonyms").warning(
        "Embeddings disabled – gensim not importable (%s). "
        'Run  pip install "gensim>=4.3,<5"  to enable.',
        _e,
    )

# ── Stop‑words & NLTK bootstrap ───────────────────────────────────────────────
_BASIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "he", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "to", "was", "were",
    "will", "with",
}

def ensure_nltk_corpora() -> None:
    for corpus in ("wordnet", "omw-1.4"):
        try:
            __import__(f"nltk.corpus.{corpus.replace('-', '_')}")
        except (ImportError, LookupError):
            nltk.download(corpus, quiet=True)

ensure_nltk_corpora()

# ── Logging -------------------------------------------------------------------
LOGGER = logging.getLogger("synonyms")
LOGGER.setLevel(logging.INFO)
_hdlr = logging.StreamHandler()
_hdlr.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOGGER.addHandler(_hdlr)

# ── Text helpers --------------------------------------------------------------
_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
lemmatizer = WordNetLemmatizer()


def _normalise_token(tok: str) -> str:
    clean = _WORD_RE.sub("", tok.strip().lower())
    return lemmatizer.lemmatize(clean)


def _acceptable(term: str) -> bool:
    return len(term) > 1 and term not in _BASIC_STOPWORDS


async def _retry_async(
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    attempts: int = 3,
    delay: float = 0.3,
    backoff: float = 2.0,
    name: str = "operation",
) -> Any:
    exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except Exception as e:  # pragma: no cover
            exc = e
            LOGGER.warning("%s attempt %d/%d failed: %s", name, attempt, attempts, e)
            if attempt < attempts:
                await asyncio.sleep(delay)
                delay *= backoff
    LOGGER.error("%s ultimately failed after %d attempts: %s", name, attempts, exc)
    return None


# ── Core class ----------------------------------------------------------------
class SynonymGenerator:
    DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
    _EMBED_MODEL_CACHE: Optional["KeyedVectors"] = None

    def __init__(
        self,
        *,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        embedding_model_path: str | None = None,
        embedding_topn: int = 10,
    ) -> None:
        self.ollama_base_url = ollama_base_url or self.DEFAULT_OLLAMA_BASE
        self.ollama_model = ollama_model
        self.embedding_model_path_raw = embedding_model_path or ""
        self.embedding_model_path = Path(embedding_model_path) if embedding_model_path else None
        self.embedding_topn = embedding_topn
        self._client: AsyncClient | None = None

        self.available_methods: Dict[str, Callable[[List[str]], Coroutine[Any, Any, Dict[str, Any]]]] = {
            "wordnet": self._generate_wordnet,
            "embeddings": (
                self._generate_embeddings if _EMBEDDINGS_AVAILABLE else self._embeddings_unavailable
            ),
            "llm": self._generate_llm if _LLM_AVAILABLE else self._llm_unavailable,
        }

    # ---------------------------------------------------------------- public API
    async def generate(
        self,
        inp: str | List[str],
        methods: List[str] | None = None,
    ) -> Dict[str, Any]:
        tokens = self._input_to_tokens(inp)
        result: Dict[str, Any] = {"input": tokens, "methods": {}, "errors": {}}

        chosen = methods or list(self.available_methods)
        LOGGER.info("Using synonym methods: %s", ", ".join(chosen))

        overall_t0 = time.perf_counter()

        for m in chosen:
            if m not in self.available_methods:
                LOGGER.warning("Method %s not implemented; skipping", m)
                result["errors"][m] = "not-implemented"
                continue
            LOGGER.info("Running synonym method: %s", m)
            t0 = time.perf_counter()
            try:
                syns = await self.available_methods[m](tokens)
                dt = time.perf_counter() - t0
                result["methods"][m] = syns
                total = sum(len(v) for v in syns.values())
                LOGGER.info("Method %s produced %d synonyms in %.2fs", m, total, dt)
            except Exception as e:
                dt = time.perf_counter() - t0
                LOGGER.exception("Method %s failed after %.2fs", m, dt)
                result["errors"][m] = str(e)

        LOGGER.info("All methods completed in %.2fs", time.perf_counter() - overall_t0)
        return result

    # ----------------------------------------------------------- wordnet method
    @staticmethod
    def _lemma_weight(lemma) -> int:
        return getattr(lemma, "count", lambda: 1)()

    async def _generate_wordnet(self, tokens: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for tok in tokens:
            counts: Counter[str] = Counter()
            for syn in wordnet.synsets(tok):
                for lemma in syn.lemmas():
                    term = _normalise_token(lemma.name().replace("_", " "))
                    if term and term != tok and _acceptable(term):
                        counts[term] += self._lemma_weight(lemma)

            if not counts:
                out[tok] = []
                continue

            max_c = max(counts.values())
            sorted_terms: List[Tuple[str, int]] = counts.most_common()
            out[tok] = [
                {"term": term, "weight": round(count / max_c, 4)}
                for term, count in sorted_terms
            ]
        return out

    # -------------------------------------------------------- embeddings method
    async def _generate_embeddings(self, tokens: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        model = await self._load_embeddings()
        if model is None:
            LOGGER.warning("Embeddings model unavailable – skipping")
            return {tok: [] for tok in tokens}

        out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for tok in tokens:
            if tok not in model.key_to_index:
                out[tok] = []
                continue
            seen: set[str] = set()
            for word, score in model.most_similar(tok, topn=self.embedding_topn * 2):
                clean = _normalise_token(word)
                if not _acceptable(clean) or clean in seen or clean == tok:
                    continue
                out[tok].append({"term": clean, "weight": round(float(score), 4)})
                seen.add(clean)
                if len(out[tok]) >= self.embedding_topn:
                    break
        return out

    async def _embeddings_unavailable(
        self, tokens: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        LOGGER.error(
            "Embeddings method skipped – gensim is not installed.\n"
            'Fix:  pip install "gensim>=4.3,<5"'
        )
        return {tok: [] for tok in tokens}

    async def _load_embeddings(self) -> Optional["KeyedVectors"]:
        if KeyedVectors is None or api is None:
            return None

        if self.__class__._EMBED_MODEL_CACHE is not None:
            return self.__class__._EMBED_MODEL_CACHE

        # local path?
        if self.embedding_model_path and self.embedding_model_path.exists():
            try:
                self.__class__._EMBED_MODEL_CACHE = KeyedVectors.load(
                    str(self.embedding_model_path), mmap="r"
                )
                return self._EMBED_MODEL_CACHE
            except Exception:
                try:
                    self.__class__._EMBED_MODEL_CACHE = KeyedVectors.load_word2vec_format(
                        str(self.embedding_model_path),
                        binary=self.embedding_model_path.suffix == ".bin",
                    )
                    return self._EMBED_MODEL_CACHE
                except Exception as e:
                    LOGGER.error("Local embedding load failed: %s", e)
                    return None

        # gensim‑data
        model_name = self.embedding_model_path_raw or "glove-wiki-gigaword-300"
        self.__class__._EMBED_MODEL_CACHE = api.load(model_name)
        return self._EMBED_MODEL_CACHE

    # ----------------------------------------------------------------------- LLM
    async def _llm_unavailable(
        self, tokens: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        LOGGER.error(
            "LLM method skipped –llama client not installed. "
            "Install `ollama` and ensure an Ollama server is reachable."
        )
        return {tok: [] for tok in tokens}

    async def _generate_llm(self, tokens: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not await self._ensure_ollama_ready():
            LOGGER.error("Ollama unavailable")
            return {tok: [] for tok in tokens}

        prompt = (
            "Return a verbose expanded dict of synonyms based on the following keywords - ONLY return JSON:\n"
            '{ "expanded": { "keyword": ["syn1","syn2", ...] } }\n\n'
            f"Keywords: {json.dumps(tokens)}"
        )

        async def call_chat():
            assert self._client  # nosec B101
            return await self._client.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.4},
            )

        resp = await _retry_async(call_chat, name="ollama.chat")
        if resp is None:
            return {tok: [] for tok in tokens}

        raw = resp["message"]["content"].strip("` \n")
        try:
            expanded = json.loads(raw).get("expanded", {})
        except Exception as e:
            LOGGER.warning("Failed to parse LLM output: %s\nRaw content: %r", e, raw)
            expanded = {}

        LOGGER.debug("LLM expanded keys: %s", list(expanded))

        # Post‑process: filter & weight.
        model = await self._load_embeddings()  # may be None
        out = {}
        for tok in tokens:
            cleaned: List[str] = []
            seen: set[str] = set()
            for term in expanded.get(tok, []):
                normal = _normalise_token(term)
                if _acceptable(normal) and normal not in seen and normal != tok:
                    cleaned.append(normal)
                    seen.add(normal)

            scored: List[Dict[str, Any]] = []
            if model and tok in model.key_to_index:
                for term in cleaned:
                    weight = round(float(model.similarity(tok, term)), 4) if term in model.key_to_index else 0.0
                    scored.append({"term": term, "weight": weight})
                scored.sort(key=lambda d: d["weight"], reverse=True)
            else:
                for idx, term in enumerate(cleaned):
                    scored.append({"term": term, "weight": round(max(1.0 - idx * 0.1, 0.1), 2)})

            out[tok] = scored
        return out

    async def _ensure_ollama_ready(self) -> bool:
        if AsyncClient is None:
            return False
        if self._client is None:
            self._client = AsyncClient(host=self.ollama_base_url)
        try:
            await self._client.list()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _input_to_tokens(inp: str | List[str]) -> List[str]:
        if isinstance(inp, str):
            tokens = [_normalise_token(t) for t in inp.split()]
        else:
            tokens = [_normalise_token(t) for t in inp if isinstance(t, str) and t.strip()]
        return [t for t in tokens if t]


# ── CLI (simplified – one‑shot) ----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synonym generator CLI (v3.5)")
    parser.add_argument("text", help="Input text or comma‑separated tokens")
    parser.add_argument("-m", "--methods", help="Comma‑separated list (default: all)")
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument("--embedding-model")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    if "," in args.text and not args.text.strip().endswith(("]", "}")):
        in_tokens: str | List[str] = [t.strip() for t in args.text.split(",")]
    else:
        in_tokens = args.text

    gen = SynonymGenerator(
        ollama_model=args.ollama_model,
        embedding_model_path=args.embedding_model,
        embedding_topn=args.topn,
    )

    methods = [m.strip() for m in args.methods.split(",")] if args.methods else list(
        gen.available_methods
    )

    result = asyncio.run(gen.generate(in_tokens, methods))
    print(json.dumps(result, indent=2, ensure_ascii=False))
