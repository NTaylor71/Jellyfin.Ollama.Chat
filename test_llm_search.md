# 🎬 Semantic Movie Search System (LLM + FAISS)

This repository contains a powerful yet beginner-friendly movie search system built with Python. It allows you to search your local movie collection using natural language queries like:

> "films that were banned for political reasons"

> "slow-burn thrillers with unreliable narrators"

Instead of using traditional literal keyword matching, this system:

- Uses Large Language Models (LLMs) to **understand and enrich** your search query
- Embeds movie data into vectors using semantic meaning
- Uses FAISS (Facebook AI Similarity Search) to **quickly find similar movies** based on meaning, not just keywords
- Boosts results if keyword matches are found in key areas like censorship, genre commentary, or summary

It provides full-text explanations and shows you **why** results matched (inline explanations).

This README explains every component, file, and step in simple terms — ideal for junior devs, self-learners, or AI-curious Python programmers.

---

## 🧠 How It Works (Conceptually)

Imagine you want to find movies "that got people arrested". Your search input is vague. This system makes it smart by doing the following:

1. **Prompt Enrichment**: We ask an LLM (like LLaMA or Mistral) to rewrite your vague query into a detailed description — like a paragraph. It also pulls out keywords like `arrest`, `censorship`, `protest`.

2. **Movie Enrichment**: Each movie description (like title + plot) is also passed through the LLM to extract rich metadata fields like:

   - `summary`
   - `censorship`
   - `genre_commentary`
   - `viewer_summaries`
   - `historical_context`

3. **Embedding**: The enriched movie data and your enriched query paragraph are converted into numbers — high-dimensional **vectors** that capture meaning.

4. **FAISS Similarity Search**: These vectors are stored in a fast index (`faiss.index`). We search for the closest movie vectors to your query vector.

5. **Keyword Boosting**: After retrieving the top matches, we scan them for keyword matches in important fields. Matches get a score bump and a friendly explanation like:

   > 💬 censorship: "banned", summary: "arrest"

6. **Ranking + Results**: The system shows the top results, sorted by semantic similarity + keyword match boosts, with full explanations.

---

## 🗂️ Key Files and Their Roles

### 🧠 `LLMPromptEnricherPlugin`

- Takes your query
- Uses an LLM to return:
  ```json
  {
    "query_paragraph": "A verbose version of your query...",
    "keywords": ["banned", "censorship"]
  }
  ```
- Used to create a **semantic search anchor** and keyword boost logic

### 🎬 `LLMEnricherPlugin`

- Takes each movie block and returns structured JSON:
  ```json
  {
    "summary": "...",
    "censorship": "...",
    "genre_commentary": "...",
    ...
  }
  ```
- Also creates a `full_text` field (human readable) used for embedding

### 🧠 `CPUEmbedderPlugin`

- Uses an Ollama-hosted embedding model (like `nomic-embed-text`) to turn strings into vectors
- Used for both movies and queries

### 🔍 `test_llm_search.py`

Main script that:

- Loads or enriches movies
- Embeds and indexes them with FAISS
- Accepts a query from CLI
- Enriches the query
- Finds top matches by vector similarity
- Applies keyword boosts and explanations
- Prints results with:
  - Score
  - Matched keywords by field
  - Final verdict

---

## 💾 Caching & Persistence

| File                  | Purpose                                                  |
| --------------------- | -------------------------------------------------------- |
| `embedding_cache.pkl` | Stores enriched entries: `full_text`, `fields`, `vector` |
| `faiss.index`         | Binary FAISS vector index (fast reload)                  |

---

## ✅ CLI Usage Examples

```bash
python test_llm_search.py \
  --query "films that were banned for social commentary" \
  --top-n 5 \
  --keyword-weight 0.1 \
  --query-mode verbose \
  --show-enriched \
  --debug-query
```

---

## ⚙️ Flags Explained

| Flag               | What it does                                  |
| ------------------ | --------------------------------------------- |
| `--query`          | Your search string                            |
| `--top-n`          | How many results to return                    |
| `--keyword-weight` | Boost factor for each matched keyword         |
| `--query-mode`     | `literal`, `verbose`, or `hybrid` blending    |
| `--skip-embed`     | Use cache instead of re-embedding everything  |
| `--show-enriched`  | Print full enriched movie text (can be long!) |
| `--debug-query`    | Print the enriched query paragraph + keywords |

---

## 🧠 Why Use This?

- Unlike keyword search, this understands concepts like:
  - *"banned for sexual content"*, *"films that critique capitalism"*
- Learns fuzzy associations (e.g., "arrested" ≈ "banned" or "censored")
- Transparent results — tells you **why** it matched
- Runs locally using Ollama + FAISS — no API keys or OpenAI billing
- Easy to customize or extend per field, source, or LLM

---

## 🛠️ Prerequisites

- Python 3.10+
- Ollama running a model with chat + embedding (e.g. `llama3`, `nomic-embed-text`)
- Install deps:
  ```bash
  pip install faiss-cpu colorama numpy
  ```

---

## 🧪 Next Ideas (For You!)

- Add genre-based filtering (e.g., `--require-tag drama`)
- Export matches to JSONL or web dashboard
- Support multilingual queries + subtitles
- Use different keyword weights per field
- Add UI using Streamlit or FastAPI

---

## ❤️ Thanks

This project is designed for curious developers learning LLMs, vector search, and natural language processing. Whether you're a junior engineer or building your first AI tool — welcome.

You're encouraged to fork, improve, and expand it!

---

Happy hacking 🎬

