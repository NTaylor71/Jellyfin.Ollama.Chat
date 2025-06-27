# /src/data/sample_entries.py

from uuid import uuid4
from src.config import FAISS_VECTOR_DIM


SAMPLE_VECTORS = [
    {
        "id": str(uuid4()),
        "vector": [0.1 * i for i in range(FAISS_VECTOR_DIM)],
        "metadata": {
            "title": "The Matrix",
            "year": 1999,
            "genres": ["Science Fiction", "Action"],
            "tagline": "Welcome to the Real World",
            "overview": "A hacker discovers the world is a simulation...",
            "actors": ["Keanu Reeves", "Laurence Fishburne"],
            "certificate": "R",
            "media_type": "Movie",
            "language": "English"
        }
    },
    {
        "id": str(uuid4()),
        "vector": [0.2 * i for i in range(FAISS_VECTOR_DIM)],
        "metadata": {
            "title": "Pulp Fiction",
            "year": 1994,
            "genres": ["Crime", "Drama"],
            "tagline": "Just because you are a character doesn't mean you have character.",
            "overview": "The lives of two mob hitmen, a boxer, and others intertwine.",
            "actors": ["John Travolta", "Samuel L. Jackson"],
            "certificate": "R",
            "media_type": "Movie",
            "language": "English"
        }
    }
]

def get_sample_entries():
    sample_entries = [
        {
            "title": "The Matrix",
            "year": 1999,
            "genres": ["Sci-Fi", "Action"],
            "overview": "A hacker discovers the world is a simulation.",
            "actors": ["Keanu Reeves", "Laurence Fishburne"],
            "tagline": "Welcome to the Real World",
            "certificate": "R",
            "media_type": "Movie",
            "language": "English",
        },
        {
            "title": "Pulp Fiction",
            "year": 1994,
            "genres": ["Crime", "Drama"],
            "overview": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
            "actors": ["John Travolta", "Uma Thurman", "Samuel L. Jackson"],
            "tagline": "Just because you are a character doesn't mean you have character.",
            "certificate": "R",
            "media_type": "Movie",
            "language": "English",
        }
    ]

    # Ensure each entry has a 'document' field
    for entry in sample_entries:
        entry["document"] = f"{entry['title']} ({entry['year']})"

    return sample_entries


def get_sample_vectors():
    """Full ingestion entries with vectors."""
    return SAMPLE_VECTORS


def get_formatter_sample():
    """Only the rich metadata of the first movie for formatter tests."""
    return SAMPLE_VECTORS[0]["metadata"]
