from src.rag.formatter import render_media_text_block
from src.data.sample_entries import get_formatter_sample
from pprint import pprint

sample = get_formatter_sample()

pprint(render_media_text_block(sample))
