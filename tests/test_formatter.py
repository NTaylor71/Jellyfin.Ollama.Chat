from src.rag.formatter import render_media_text_block
from pprint import pprint

sample = {
    "title": "The Matrix",
    "year": 1999,
    "genres": ["Science Fiction", "Action"],
    "tagline": "Welcome to the Real World",
    "overview": "A hacker discovers the world is a simulation...",
    "actors": ["Keanu Reeves", "Laurence Fishburne"],
    "certificate": "R",
    "media_type": "Movie",
    "language": "English",
}

pprint(render_media_text_block(sample))
