from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict, List

# Setup Jinja2 environment and load the template
template_dir = Path(__file__).parent
env = Environment(
    loader=FileSystemLoader(template_dir),
    trim_blocks=True,
    lstrip_blocks=True
)
template = env.get_template("media_template.j2")


def render_media_text_block(entry: Dict) -> str:
    """Render Jellyfin metadata directly into Jinja2 template."""
    return template.render(**entry).strip()

def render_entries_for_embedding(entries: List[Dict]) -> List[str]:
    """Render a list of Jellyfin metadata entries to rich text blocks."""
    return [render_media_text_block(entry) for entry in entries]
