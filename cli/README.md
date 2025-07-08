# CLI Plugin Test Scripts

This directory contains CLI test scripts for each plugin, using the queue system for real-world validation.

## Available Scripts

### Keyword Expansion Plugins
- `conceptnet_keyword_endpoint_test.py` - Test ConceptNet keyword expansion
- `llm_keyword_endpoint_test.py` - Test LLM keyword expansion
- `gensim_similarity_endpoint_test.py` - Test Gensim similarity matching
- `merge_keywords_endpoint_test.py` - Test keyword merging from multiple providers

### Temporal Analysis Plugins
- `spacy_temporal_endpoint_test.py` - Test SpaCy temporal analysis
- `heideltime_temporal_endpoint_test.py` - Test HeidelTime temporal analysis
- `sutime_temporal_endpoint_test.py` - Test SUTime temporal analysis
- `llm_temporal_intelligence_endpoint_test.py` - Test LLM temporal intelligence

### Question Answering Plugins
- `llm_question_answer_endpoint_test.py` - Test LLM question answering

## Usage

Each script follows the same pattern:

```bash
# Basic usage
python script_name.py "input_data"

# Show queue statistics
python script_name.py --stats

# Show task definition
python script_name.py --definition

# Verbose output
python script_name.py "input_data" --verbose

# Custom timeout
python script_name.py "input_data" --timeout 120
```

## Examples

### Keyword Expansion
```bash
# Test ConceptNet expansion
python conceptnet_keyword_endpoint_test.py "science fiction"

# Test LLM expansion with context
python llm_keyword_endpoint_test.py "psychological thriller" --media-context movie

# Test Gensim similarity
python gensim_similarity_endpoint_test.py "adventure" --similarity-threshold 0.7

# Test keyword merging
python merge_keywords_endpoint_test.py "horror" --providers conceptnet llm --merge-strategy weighted
```

### Temporal Analysis
```bash
# Test SpaCy temporal analysis
python spacy_temporal_endpoint_test.py "Released in 1995, set in the 1980s"

# Test HeidelTime with document date
python heideltime_temporal_endpoint_test.py "The movie came out last summer" --document-date 2023-01-01

# Test LLM temporal intelligence
python llm_temporal_intelligence_endpoint_test.py "A story about the Cold War era" --analysis-type periods
```

### Question Answering
```bash
# Test LLM question answering
python llm_question_answer_endpoint_test.py "What are the themes of Blade Runner?" --context "Blade Runner is a 1982 science fiction film"
```

## Requirements

- Redis server running
- Worker process running
- All required services (NLP, LLM, etc.) available
- Proper environment configuration

## Notes

- All scripts use the actual Redis queue system for real-world testing
- No mocks or hard-coded responses - full integration testing
- Scripts will wait for worker to process tasks and return results
- Use `--stats` to check queue health before running tests
- Use `--definition` to see what data each plugin expects