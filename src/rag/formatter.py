from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict

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

    return template.render(**entry).strip()
