import requests
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
import unicodedata


# 📦 Load environment variables from .env
load_dotenv()

JELLYFIN_URL = "https://jelly.custard.solutions:30015"
API_KEY = "cbaac61ac6404afcaeffacffde3e4242"

'''
# --------------------
# Config
# --------------------
JELLYFIN_URL = "https://jelly.custard.solutions:30015"
API_KEY = "cbaac61ac6404afcaeffacffde3e4242"
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "X-Emby-Token": API_KEY
}

params = {
    "IncludeItemTypes": "Movie",
    "Recursive": "true",
    "Fields": "Overview,Path,MediaSources,Genres,Studios,Tags,DateCreated,SortName,RunTimeTicks,ProductionYear,OfficialRating,CommunityRating,CriticRating,RemoteTrailers",
    "Limit": 500
}

# --------------------
# Fetch data
# --------------------
response = requests.get(f"{JELLYFIN_URL}/Items", headers=headers, params=params)
response.raise_for_status()
items = response.json().get("Items", [])

'''


def fetch_movies(limit=1000, query=None):
    url = f"{JELLYFIN_URL}/Items"
    headers = {"X-Emby-Token": API_KEY}
    params = {
        "IncludeItemTypes": "Movie",
        "Recursive": "true",
        "Fields": "Overview,Genres,Taglines,People,OfficialRating,ProductionYear,Name,MediaStreams",
        "Limit": limit,
    }
    if query:
        params["SearchTerm"] = query

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["Items"]


def to_ascii(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def simplify_movie(item):
    return {
        "title": to_ascii(item.get("Name", "")),
        "year": item.get("ProductionYear"),
        "genres": [to_ascii(g) for g in item.get("Genres", [])],
        "overview": to_ascii(item.get("Overview", "")),
        "tagline": to_ascii(item.get("Taglines", [""])[0]) if item.get("Taglines") else "",
        "certificate": to_ascii(item.get("OfficialRating", "")),
        "actors": [to_ascii(p["Name"]) for p in item.get("People", []) if p.get("Type") == "Actor"],
        "media_type": "Movie",
        "language": "English",  # TODO: Extract from MediaStreams?
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000, help="Max number of movies to fetch")
    parser.add_argument("--query", type=str, help="Optional search term (literal match)")
    parser.add_argument("--save", action="store_true", help="Write results to jellyfin_movies.json")
    args = parser.parse_args()

    print(f"🎬 Connecting to {JELLYFIN_URL}...")
    try:
        items = fetch_movies(limit=args.limit, query=args.query)
        simplified = [simplify_movie(item) for item in items]

        print(f"✅ Retrieved {len(simplified)} movies")
        print(json.dumps(simplified, indent=2))

        if args.save:
            with open("jellyfin_movies.json", "w", encoding="utf-8") as f:
                json.dump(simplified, f, indent=2)
            print("💾 Saved to jellyfin_movies.json")

    except Exception as e:
        print("❌ Failed to fetch or process movies:", e)

if __name__ == "__main__":
    main()
