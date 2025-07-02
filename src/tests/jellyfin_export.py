#!/usr/bin/env python3
"""
jellyfin_export.py  –  “maximal” Movie metadata export with chunking + ETA

• Pulls every ItemFields value recognised by Jellyfin (see CONFIG["query"]["Fields"]).
• Adds optional image / user‑data switches (true by default).
• Two output modes:
      – raw   – return the complete BaseItemDto objects exactly as sent by the API
      – clean – return a summary defined in CONFIG["output"]
• Batches /Items fetches into 100-item chunks with ETA and runtime stats.
"""

from __future__ import annotations
import argparse
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Union
import time

import requests
from dotenv import load_dotenv

load_dotenv()  # .env can hold JELLYFIN_URL and JELLYFIN_API_KEY

# ──────────────────────────────────────────────────────────────
# 1. CONFIGURATION – edit here
# ──────────────────────────────────────────────────────────────
ITEM_FIELDS_ALL: List[str] = [
    "AirTime", "CanDelete", "CanDownload", "ChannelInfo", "Chapters", "Trickplay",
    "ChildCount", "CumulativeRunTimeTicks", "CustomRating", "DateCreated",
    "DateLastMediaAdded", "DisplayPreferencesId", "Etag", "ExternalUrls", "Genres",
    "ItemCounts", "MediaSourceCount", "MediaSources", "OriginalTitle", "Overview",
    "ParentId", "Path", "People", "PlayAccess", "ProductionLocations", "ProviderIds",
    "PrimaryImageAspectRatio", "RecursiveItemCount", "Settings", "SeriesStudio",
    "SortName", "SpecialEpisodeNumbers", "Studios", "Taglines", "Tags",
    "RemoteTrailers", "MediaStreams", "SeasonUserData", "DateLastRefreshed",
    "DateLastSaved", "RefreshState", "ChannelImage", "EnableMediaSourceDisplay",
    "Width", "Height", "ExtraIds", "LocalTrailerCount", "IsHD", "SpecialFeatureCount",
]

CONFIG: Dict[str, Any] = {
    "server": {
        "url": os.getenv("JELLYFIN_URL", "https://jelly.custard.solutions:30015"),
        "api_key": os.getenv("JELLYFIN_API_KEY", "CHANGE_ME"),
    },
    "query": {
        "IncludeItemTypes": ["Movie"],
        "Recursive": True,
        "Fields": ITEM_FIELDS_ALL,
        "Limit": 10_000,
        "EnableImages": True,
        "EnableUserData": True,
    },
    "output_mode": "raw",
    "output": {
        "id":          {"field": "Id"},
        "title":       {"field": "Name",      "transform": "ascii"},
        "year":        {"field": "ProductionYear"},
        "genres":      {"field": "Genres",    "transform": "ascii_list"},
        "tagline":     {"field": "Taglines",  "index": 0, "transform": "ascii"},
        "overview":    {"field": "Overview",  "transform": "ascii"},
        "actors":      {"people_type": "Actor", "transform": "ascii_list"},
        "cert":        {"field": "OfficialRating", "transform": "ascii"},
        "runtime":     {"field": "RunTimeTicks"},
        "path":        {"field": "Path"},
        "provider_ids":{"field": "ProviderIds"},
        "media":       {"field": "MediaStreams"},
    },
    "chunk_size": 100,
}

# ──────────────────────────────────────────────────────────────
# 2. UTILITY
# ──────────────────────────────────────────────────────────────
def to_ascii(txt: Union[str, None]) -> str:
    if not txt:
        return ""
    return unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")

def ascii_list(vals: List[str] | None) -> List[str]:
    return [to_ascii(v) for v in vals or []]

TRANSFORM = {"ascii": to_ascii, "ascii_list": ascii_list, "identity": lambda x: x}

def flatten_params(src: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in src.items():
        if v is None:
            continue
        if isinstance(v, list):
            out[k] = ",".join(str(i) for i in v)
        elif isinstance(v, bool):
            out[k] = str(v).lower()
        else:
            out[k] = str(v)
    return out

# ──────────────────────────────────────────────────────────────
# 3. NETWORK (chunk-aware)
# ──────────────────────────────────────────────────────────────
def fetch_items(limit: int | None, search: str | None) -> List[Dict[str, Any]]:
    chunk_size = CONFIG.get("chunk_size", 100)
    total = limit or CONFIG["query"].get("Limit", 1000)
    total_chunks = (total + chunk_size - 1) // chunk_size

    all_items = []
    times = []

    for chunk_index in range(total_chunks):
        start_index = chunk_index * chunk_size
        q = CONFIG["query"].copy()
        q["StartIndex"] = start_index
        q["Limit"] = min(chunk_size, total - start_index)
        if search:
            q["SearchTerm"] = search

        url = CONFIG["server"]["url"].rstrip("/") + "/Items"
        headers = {"X-Emby-Token": CONFIG["server"]["api_key"]}

        t0 = time.perf_counter()
        try:
            resp = requests.get(url, headers=headers, params=flatten_params(q), timeout=60)
            resp.raise_for_status()
            chunk_items = resp.json().get("Items", [])
        except Exception as e:
            print(f"❌ Chunk {chunk_index + 1} failed (StartIndex={start_index}): {e}")
            chunk_items = []

        dt = time.perf_counter() - t0
        all_items.extend(chunk_items)
        times.append(dt)
        avg = sum(times) / len(times)
        eta = avg * (total_chunks - chunk_index - 1)

        print(f"📦 Chunk {chunk_index + 1}/{total_chunks} "
              f"→ {len(chunk_items)} items [{dt:.1f}s] | ETA: {eta:.1f}s")

    return all_items

# ──────────────────────────────────────────────────────────────
# 4. CLEAN‑MODE PROCESSING
# ──────────────────────────────────────────────────────────────
def extract_stream(item: Dict[str, Any], stype: str, field: str) -> Any:
    for s in item.get("MediaStreams", []):
        if s.get("Type") == stype:
            return s.get(field)
    return None

def summarise(item: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, rule in CONFIG["output"].items():    # type: ignore[arg-type]
        if "value" in rule:
            val = rule["value"]
        elif "field" in rule:
            val = item.get(rule["field"])
            if "index" in rule and isinstance(val, list):
                idx = rule["index"]
                val = val[idx] if idx < len(val) else None
        elif "people_type" in rule:
            val = [p["Name"] for p in item.get("People", []) if p.get("Type") == rule["people_type"]]
        elif "from_stream" in rule:
            spec = rule["from_stream"]
            val = extract_stream(item, spec["type"], spec["field"])
        else:
            val = None
        fn = TRANSFORM.get(rule.get("transform", "identity"), lambda x: x)
        summary[key] = fn(val)
    return summary

# ──────────────────────────────────────────────────────────────
# 5. CLI / MAIN
# ──────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Export maximal movie metadata from Jellyfin")
    ap.add_argument("--limit", type=int, help="max items (overrides CONFIG)")
    ap.add_argument("--query", type=str, help="SearchTerm filter")
    ap.add_argument("--raw", action="store_true", help="Force raw mode (ignores CONFIG)")
    ap.add_argument("-o", "--out", default="jellyfin_movies_full.json",
                    help="destination (use '-' for stdout)")
    args = ap.parse_args()

    out_mode = "raw" if args.raw else CONFIG["output_mode"]
    items = fetch_items(args.limit, args.query)

    payload = items if out_mode == "raw" else [summarise(i) for i in items]

    txt = json.dumps(payload, indent=2)
    if args.out == "-":
        print(txt)
    else:
        Path(args.out).write_text(txt, encoding="utf-8")
        print(f"\n✅ Wrote {len(payload)} records ➜ {args.out}")

if __name__ == "__main__":
    main()
