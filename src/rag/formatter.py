from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict
from src.data.sample_entries import get_sample_entries  # Import the function

# Load Jinja2 environment
template_dir = Path(__file__).parent
env = Environment(
    loader=FileSystemLoader(template_dir),
    trim_blocks=True,
    lstrip_blocks=True
)
template = env.get_template("media_template.j2")

def render_media_text_block(entry: Dict) -> str:
    """Render Jellyfin metadata as a rich text block for embedding."""
    # Ensure that we have all necessary fields in the entry
    entry = {
        "title": entry.get("title", "Unknown Title"),
        "year": entry.get("year", "Unknown Year"),
        "genres": entry.get("genres", []),
        "tagline": entry.get("tagline", ""),
        "overview": entry.get("overview", ""),
        "actors": entry.get("actors", []),
        "certificate": entry.get("certificate", ""),
        "media_type": entry.get("media_type", "Movie"),
        "language": entry.get("language", "English"),
    }

    # Render the template using the movie metadata
    return template.render(**entry).strip()

# Fetch the actual movie data from sample_entries.py
sample_entries = get_sample_entries()

# Example: Render the metadata for the first movie in the list
rendered_text = render_media_text_block(sample_entries[0])
print(rendered_text)
