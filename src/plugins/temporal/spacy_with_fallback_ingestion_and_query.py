"""
spaCy-based dual-use temporal analysis plugin for both ingestion and query processing.
Provides sophisticated temporal understanding with fallback strategies.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Core NLP libraries
try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError as e:
    SPACY_AVAILABLE = False
    logging.getLogger(__name__).debug(f"spaCy not available: {e}. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from dateutil.parser import parse as dateutil_parse
    from dateutil.relativedelta import relativedelta
    DATEUTIL_AVAILABLE = True
except ImportError as e:
    DATEUTIL_AVAILABLE = False
    logging.getLogger(__name__).debug(f"python-dateutil not available: {e}. Install with: pip install python-dateutil")

try:
    import arrow
    ARROW_AVAILABLE = True
except ImportError as e:
    ARROW_AVAILABLE = False
    logging.getLogger(__name__).debug(f"arrow not available: {e}. Install with: pip install arrow")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logging.getLogger(__name__).debug(f"transformers not available: {e}. Install with: pip install transformers torch")

from src.plugins.linguistic.base import DualUsePlugin


class SpacyWithFallbackIngestionAndQueryPlugin(DualUsePlugin):
    """
    Dual-use spaCy-based temporal analysis for both content ingestion and user queries.
    
    INGESTION MODE: Analyzes movie content for temporal metadata storage
    QUERY MODE: Processes user queries for temporal search enhancement
    
    Uses spaCy + transformers + dateutil + arrow with sophisticated fallback strategies.
    Provides Google 2010-level temporal understanding for both use cases.
    """
    
    def __init__(self):
        super().__init__()
        self.current_year = datetime.now().year
        self.current_date = datetime.now()
        
        # Initialize NLP models
        self.nlp = None
        self.matcher = None
        self.ner_pipeline = None
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the dual-use temporal plugin."""
        try:
            self._initialize_models()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize spaCy temporal plugin: {e}")
            return False
    
    async def embellish_query(self, query: str, context) -> str:
        """QUERY MODE: Embellish user query with temporal analysis."""
        try:
            # Perform temporal analysis on the query
            analysis = await self.analyze(query, context.metadata if hasattr(context, 'metadata') else None)
            
            # For now, return the original query (could expand with temporal terms)
            # In future iterations, we could expand queries like "90s movies" to "1990s movies"
            return query
            
        except Exception as e:
            self.logger.error(f"Error in temporal query embellishment: {e}")
            return query
    
    async def embellish_embed_data(self, data: Dict[str, Any], context) -> Dict[str, Any]:
        """INGESTION MODE: Embellish content data with temporal analysis."""
        try:
            # Extract text for analysis
            text_content = ""
            if isinstance(data, dict):
                # Extract text from common fields
                for field in ['overview', 'description', 'plot', 'summary', 'name', 'title']:
                    if field in data and data[field]:
                        text_content += f" {data[field]}"
            else:
                text_content = str(data)
            
            if text_content.strip():
                context_dict = context.metadata if hasattr(context, 'metadata') else None
                temporal_analysis = await self.analyze(text_content.strip(), context_dict)
                
                # Add temporal analysis to enhanced_fields for ingestion storage
                if 'enhanced_fields' not in data:
                    data['enhanced_fields'] = {}
                
                data['enhanced_fields']['spacy_temporal_analysis'] = temporal_analysis
                
                # Extract useful temporal metadata for search
                if temporal_analysis.get('normalized'):
                    temporal_metadata = {
                        'detected_time_periods': [],
                        'temporal_scope': temporal_analysis.get('temporal_scope', 'unknown'),
                        'confidence': temporal_analysis.get('confidence_level', 'medium'),
                        'analysis_methods': temporal_analysis.get('analysis_methods', [])
                    }
                    
                    for norm in temporal_analysis['normalized']:
                        temporal_metadata['detected_time_periods'].append({
                            'text': norm.get('text'),
                            'start': norm.get('start'),
                            'end': norm.get('end'),
                            'precision': norm.get('precision'),
                            'method': norm.get('method', 'unknown')
                        })
                    
                    data['enhanced_fields']['spacy_temporal_metadata'] = temporal_metadata
                    
                    # Create searchable temporal tags for ingestion
                    search_tags = self._create_ingestion_search_tags(temporal_analysis)
                    if search_tags:
                        data['enhanced_fields']['spacy_temporal_search_tags'] = search_tags
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in temporal embellishment: {e}")
            return data
        
    def _initialize_models(self):
        """Initialize sophisticated NLP models."""
        available_methods = []
        
        try:
            # Initialize spaCy
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.matcher = Matcher(self.nlp.vocab)
                    self._setup_temporal_patterns()
                    self.logger.info("✅ spaCy temporal model loaded successfully")
                    available_methods.append("spacy")
                except OSError as e:
                    self.logger.info(f"📦 spaCy model not found, attempting automatic download...")
                    try:
                        import subprocess
                        import sys
                        # Attempt automatic model download
                        result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                                              capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            self.logger.info("✅ spaCy model downloaded successfully, loading...")
                            self.nlp = spacy.load("en_core_web_sm")
                            self.matcher = Matcher(self.nlp.vocab)
                            self._setup_temporal_patterns()
                            self.logger.info("✅ spaCy temporal model loaded successfully")
                            available_methods.append("spacy")
                        else:
                            self.logger.warning(f"⚠️ Failed to auto-download spaCy model: {result.stderr}")
                            self.nlp = None
                    except Exception as download_error:
                        self.logger.warning(f"⚠️ Could not auto-download spaCy model: {download_error}")
                        self.nlp = None
            else:
                self.logger.debug("📦 spaCy not available - install with: pip install spacy")
                    
            # Initialize transformers NER pipeline
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.ner_pipeline = pipeline(
                        "ner",
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        aggregation_strategy="simple"
                    )
                    self.logger.info("✅ Transformers NER pipeline loaded successfully")
                    available_methods.append("transformers")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load transformers pipeline: {e}")
                    self.ner_pipeline = None
            else:
                self.logger.debug("📦 Transformers not available - install with: pip install transformers torch")
            
            # Log dateutil availability
            if DATEUTIL_AVAILABLE:
                available_methods.append("dateutil")
                self.logger.debug("✅ python-dateutil available for sophisticated date parsing")
            else:
                self.logger.debug("📦 python-dateutil not available - install with: pip install python-dateutil")
            
            # Log arrow availability
            if ARROW_AVAILABLE:
                available_methods.append("arrow")
                self.logger.debug("✅ arrow available for relative date calculations")
            else:
                self.logger.debug("📦 arrow not available - install with: pip install arrow")
            
            # Summary logging
            if available_methods:
                self.logger.info(f"🧠 spaCy temporal analysis initialized with methods: {', '.join(available_methods)}")
            else:
                self.logger.warning("⚠️ No sophisticated temporal analysis libraries available. Install [nlp] dependencies for full functionality.")
                    
        except Exception as e:
            self.logger.error(f"💥 Error initializing temporal models: {e}")
    
    def _setup_temporal_patterns(self):
        """Setup sophisticated temporal patterns for spaCy matcher."""
        if not self.matcher:
            return
            
        try:
            # Decade patterns with context - enhanced for better detection
            decade_patterns = [
                # Standard decade patterns: 90s, 1990s, early 90s, late 2000s
                [{"LOWER": {"IN": ["early", "mid", "late"]}, "OP": "?"}, {"LIKE_NUM": True}, {"LOWER": "s"}],
                [{"LOWER": "the", "OP": "?"}, {"LIKE_NUM": True}, {"LOWER": "s"}],
                # Text-based decades: eighties, nineties
                [{"LOWER": {"IN": ["early", "mid", "late"]}, "OP": "?"}, {"LOWER": {"IN": ["eighties", "nineties", "seventies", "sixties", "fifties", "forties", "thirties", "twenties", "tens", "hundreds"]}}],
                # Combined patterns: 80s and 90s
                [{"LIKE_NUM": True}, {"LOWER": "s"}, {"LOWER": {"IN": ["and", "&"]}}, {"LIKE_NUM": True}, {"LOWER": "s"}],
                # Year ranges: 1990-2000, between 1990 and 2000
                [{"LOWER": "between"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["and", "to", "-"]}}, {"LIKE_NUM": True}],
                [{"LIKE_NUM": True}, {"LOWER": {"IN": ["-", "to"]}}, {"LIKE_NUM": True}]
            ]
            
            # Relative time patterns - enhanced
            relative_patterns = [
                # Standard relative: last decade, past decade, recent years
                [{"LOWER": {"IN": ["last", "past", "previous"]}}, {"LOWER": {"IN": ["decade", "century", "year", "years", "few"]}, "OP": "+"}, {"IS_ALPHA": True, "OP": "?"}],
                [{"LOWER": {"IN": ["recent", "lately", "nowadays"]}}, {"IS_ALPHA": True, "OP": "*"}],
                # X years/decades ago: two decades ago, 5 years ago
                [{"LOWER": {"IN": ["one", "two", "three", "four", "five", "a", "couple"]}, "OP": "?"}, {"LIKE_NUM": True, "OP": "?"}, {"LOWER": {"IN": ["years", "decades"]}}, {"LOWER": "ago"}],
                [{"LOWER": {"IN": ["this", "current"]}}, {"LOWER": {"IN": ["decade", "century", "year"]}}],
                # Post/pre patterns: post-millennium, pre-digital
                [{"LOWER": {"REGEX": r"(post|pre)-?\w+"}}],
                # Turn of century patterns
                [{"LOWER": {"IN": ["turn", "end", "beginning"]}}, {"LOWER": "of"}, {"LOWER": "the", "OP": "?"}, {"LOWER": {"IN": ["century", "millennium"]}}]
            ]
            
            # Season and period patterns - enhanced
            season_patterns = [
                # Seasonal: summer of 1993, winter 2000
                [{"LOWER": {"IN": ["spring", "summer", "fall", "autumn", "winter"]}}, {"LOWER": "of", "OP": "?"}, {"LIKE_NUM": True}],
                [{"LOWER": {"IN": ["beginning", "start", "end"]}}, {"LOWER": "of"}, {"LOWER": "the", "OP": "?"}, {"LIKE_NUM": True}, {"LOWER": "s", "OP": "?"}],
                # Around/circa patterns: around 1995, circa 2000
                [{"LOWER": {"IN": ["around", "about", "circa"]}}, {"LIKE_NUM": True}]
            ]
            
            # Add patterns to matcher with error handling
            patterns_added = 0
            if decade_patterns:
                self.matcher.add("DECADE", decade_patterns)
                patterns_added += len(decade_patterns)
                self.logger.debug(f"✅ Added {len(decade_patterns)} DECADE patterns")
            if relative_patterns:
                self.matcher.add("RELATIVE", relative_patterns)
                patterns_added += len(relative_patterns)
                self.logger.debug(f"✅ Added {len(relative_patterns)} RELATIVE patterns")
            if season_patterns:
                self.matcher.add("SEASON", season_patterns)
                patterns_added += len(season_patterns)
                self.logger.debug(f"✅ Added {len(season_patterns)} SEASON patterns")
                
            self.logger.info(f"✅ Set up {patterns_added} temporal patterns successfully")
            
        except Exception as e:
            self.logger.error(f"⚠️ Error setting up temporal patterns: {e}")
            import traceback
            self.logger.error(f"Pattern setup traceback: {traceback.format_exc()}")
            # Continue without patterns - fallback methods will still work
    
    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract temporal expressions using sophisticated NLP."""
        try:
            results = {
                "expressions": [],
                "normalized": [],
                "confidence_scores": [],
                "analysis_methods": []
            }
            
            # Method 1: spaCy NER + custom patterns
            if self.nlp:
                spacy_results = self._analyze_with_spacy(text)
                results["expressions"].extend(spacy_results["expressions"])
                results["normalized"].extend(spacy_results["normalized"])
                results["analysis_methods"].append("spacy")
            
            # Method 2: Transformers NER
            if self.ner_pipeline:
                transformer_results = self._analyze_with_transformers(text)
                results["expressions"].extend(transformer_results["expressions"])
                results["normalized"].extend(transformer_results["normalized"])
                results["analysis_methods"].append("transformers")
            
            # Method 3: dateutil parsing
            if DATEUTIL_AVAILABLE:
                dateutil_results = self._analyze_with_dateutil(text)
                results["expressions"].extend(dateutil_results["expressions"])
                results["normalized"].extend(dateutil_results["normalized"])
                results["analysis_methods"].append("dateutil")
            
            # Method 4: Arrow relative dates
            if ARROW_AVAILABLE:
                arrow_results = self._analyze_with_arrow(text)
                results["expressions"].extend(arrow_results["expressions"])
                results["normalized"].extend(arrow_results["normalized"])
                results["analysis_methods"].append("arrow")
            
            # Merge and deduplicate results
            final_results = self._merge_and_deduplicate(results)
            
            # Add semantic analysis
            final_results.update(self._semantic_temporal_analysis(text, final_results))
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in sophisticated temporal analysis: {e}")
            return {"error": str(e), "expressions": [], "normalized": [], "fallback": True}
    
    def _analyze_with_spacy(self, text: str) -> Dict[str, Any]:
        """Analyze temporal expressions using spaCy."""
        expressions = []
        normalized = []
        
        try:
            doc = self.nlp(text)
            
            # Extract DATE entities
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    expr = {
                        "text": ent.text,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "confidence": getattr(ent._, "confidence", 0.8),
                        "method": "spacy_ner"
                    }
                    expressions.append(expr)
                    
                    # Normalize the expression
                    norm = self._normalize_spacy_date(ent.text)
                    if norm:
                        normalized.append(norm)
            
            # Use custom patterns
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                label = self.nlp.vocab.strings[match_id]
                
                expr = {
                    "text": span.text,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "confidence": 0.9,
                    "method": f"spacy_pattern_{label.lower()}"
                }
                expressions.append(expr)
                
                # Normalize based on pattern type
                norm = self._normalize_spacy_pattern(span.text, label)
                if norm:
                    normalized.append(norm)
                    
        except Exception as e:
            self.logger.debug(f"spaCy analysis error: {e}")
        
        return {"expressions": expressions, "normalized": normalized}
    
    def _analyze_with_transformers(self, text: str) -> Dict[str, Any]:
        """Analyze temporal expressions using transformer models."""
        expressions = []
        normalized = []
        
        try:
            # Use NER pipeline to find entities
            entities = self.ner_pipeline(text)
            
            for entity in entities:
                if entity["entity_group"] in ["DATE", "TIME", "MISC"]:
                    expr = {
                        "text": entity["word"],
                        "start_char": entity["start"],
                        "end_char": entity["end"],
                        "confidence": entity["score"],
                        "method": "transformers_ner"
                    }
                    expressions.append(expr)
                    
                    # Try to normalize
                    norm = self._normalize_transformer_entity(entity["word"])
                    if norm:
                        normalized.append(norm)
                        
        except Exception as e:
            self.logger.debug(f"Transformers analysis error: {e}")
        
        return {"expressions": expressions, "normalized": normalized}
    
    def _analyze_with_dateutil(self, text: str) -> Dict[str, Any]:
        """Analyze temporal expressions using dateutil parser."""
        expressions = []
        normalized = []
        
        try:
            # Extract potential date phrases
            date_phrases = self._extract_date_phrases(text)
            
            for phrase in date_phrases:
                try:
                    parsed = dateutil_parse(phrase, fuzzy=True)
                    expr = {
                        "text": phrase,
                        "confidence": 0.7,
                        "method": "dateutil"
                    }
                    expressions.append(expr)
                    
                    norm = {
                        "text": phrase,
                        "start": parsed.year,
                        "end": parsed.year,
                        "precision": "year",
                        "parsed_date": parsed.isoformat(),
                        "method": "dateutil"
                    }
                    normalized.append(norm)
                    
                except Exception:
                    continue
                    
        except Exception as e:
            self.logger.debug(f"dateutil analysis error: {e}")
        
        return {"expressions": expressions, "normalized": normalized}
    
    def _analyze_with_arrow(self, text: str) -> Dict[str, Any]:
        """Analyze relative temporal expressions using arrow."""
        expressions = []
        normalized = []
        
        try:
            # Define relative patterns with arrow
            relative_mappings = {
                r'\blast\s+decade\b': lambda: arrow.now().shift(years=-10),
                r'\bpast\s+decade\b': lambda: arrow.now().shift(years=-10),
                r'\brecent\s+years?\b': lambda: arrow.now().shift(years=-3),
                r'\bthis\s+decade\b': lambda: arrow.now().replace(year=arrow.now().year // 10 * 10),
                r'\b(\d+)\s+years?\s+ago\b': lambda m: arrow.now().shift(years=-int(m.group(1))),
                r'\btwo\s+decades?\s+ago\b': lambda: arrow.now().shift(years=-20),
                r'\ba\s+decade\s+ago\b': lambda: arrow.now().shift(years=-10),
            }
            
            for pattern, calculator in relative_mappings.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        if r'(\d+)' in pattern:
                            target_date = calculator(match)
                        else:
                            target_date = calculator()
                        
                        expr = {
                            "text": match.group(0),
                            "start_char": match.start(),
                            "end_char": match.end(),
                            "confidence": 0.8,
                            "method": "arrow_relative"
                        }
                        expressions.append(expr)
                        
                        norm = {
                            "text": match.group(0),
                            "start": target_date.year,
                            "end": target_date.year,
                            "precision": "year",
                            "arrow_date": target_date.isoformat(),
                            "method": "arrow"
                        }
                        normalized.append(norm)
                        
                    except Exception:
                        continue
                        
        except Exception as e:
            self.logger.debug(f"Arrow analysis error: {e}")
        
        return {"expressions": expressions, "normalized": normalized}
    
    def _extract_date_phrases(self, text: str) -> List[str]:
        """Extract potential date phrases from text."""
        phrases = []
        
        # Common temporal phrase patterns
        patterns = [
            r'\b(?:in\s+)?(?:the\s+)?(?:early\s+|late\s+|mid\s+)?(?:19|20)\d{2}s?\b',
            r'\b(?:last|past|recent|this|current)\s+(?:few\s+)?(?:years?|decades?|century)\b',
            r'\b(?:spring|summer|fall|autumn|winter)\s+(?:of\s+)?(?:19|20)\d{2}\b',
            r'\b(?:beginning|start|end)\s+of\s+(?:the\s+)?(?:19|20)\d{2}s?\b',
            r'\b(?:between|from)\s+(?:19|20)\d{2}\s+(?:and|to)\s+(?:19|20)\d{2}\b',
            r'\b\d{1,2}\s+(?:years?|decades?)\s+ago\b',
            r'\b(?:around|about|circa)\s+(?:19|20)\d{2}\b'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrases.append(match.group(0))
        
        return phrases
    
    def _normalize_spacy_date(self, date_text: str) -> Optional[Dict[str, Any]]:
        """Normalize spaCy detected dates."""
        try:
            import re
            
            # Handle decade patterns first (most common case)
            if 's' in date_text.lower():
                decade_match = re.search(r'(\d{2,4})s', date_text)
                if decade_match:
                    year = int(decade_match.group(1))
                    if year < 100:
                        year = year + 1900 if year >= 20 else year + 2000
                    return {
                        "text": date_text,
                        "start": year,
                        "end": year + 9,
                        "precision": "decade",
                        "method": "spacy_decade"
                    }
            
            # Handle post-millennium patterns
            if "post-millennium" in date_text.lower():
                return {
                    "text": date_text,
                    "start": 2000,
                    "end": datetime.now().year,
                    "precision": "post_period",
                    "method": "spacy_event"
                }
            
            # Handle recent patterns
            if "recent" in date_text.lower():
                current_year = datetime.now().year
                return {
                    "text": date_text,
                    "start": current_year - 5,
                    "end": current_year,
                    "precision": "recent",
                    "method": "spacy_relative"
                }
            
            # Try dateutil parsing
            if DATEUTIL_AVAILABLE:
                try:
                    parsed = dateutil_parse(date_text, fuzzy=True)
                    return {
                        "text": date_text,
                        "start": parsed.year,
                        "end": parsed.year,
                        "precision": "year",
                        "method": "spacy_dateutil"
                    }
                except Exception:
                    pass
            
            return None
            
        except Exception:
            return None
    
    def _normalize_spacy_pattern(self, text: str, label: str) -> Optional[Dict[str, Any]]:
        """Normalize spaCy pattern matches."""
        try:
            if label == "DECADE":
                return self._normalize_decade_expression(text)
            elif label == "RELATIVE":
                return self._normalize_relative_expression(text)
            elif label == "SEASON":
                return self._normalize_season_expression(text)
            return None
        except Exception:
            return None
    
    def _normalize_transformer_entity(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """Normalize transformer detected entities."""
        try:
            import re
            
            # Handle decade patterns
            if 's' in entity_text.lower():
                decade_match = re.search(r'(\d{2,4})s', entity_text)
                if decade_match:
                    year = int(decade_match.group(1))
                    if year < 100:
                        year = year + 1900 if year >= 20 else year + 2000
                    return {
                        "text": entity_text,
                        "start": year,
                        "end": year + 9,
                        "precision": "decade",
                        "method": "transformer_decade"
                    }
            
            # Handle post-millennium
            if "post-millennium" in entity_text.lower():
                return {
                    "text": entity_text,
                    "start": 2000,
                    "end": datetime.now().year,
                    "precision": "post_period",
                    "method": "transformer_event"
                }
            
            # Handle recent
            if "recent" in entity_text.lower():
                current_year = datetime.now().year
                return {
                    "text": entity_text,
                    "start": current_year - 5,
                    "end": current_year,
                    "precision": "recent",
                    "method": "transformer_relative"
                }
            
            # Try dateutil parsing
            if DATEUTIL_AVAILABLE:
                parsed = dateutil_parse(entity_text, fuzzy=True)
                return {
                    "text": entity_text,
                    "start": parsed.year,
                    "end": parsed.year,
                    "precision": "year",
                    "method": "transformer_dateutil"
                }
        except Exception:
            pass
        return None
    
    def _normalize_decade_expression(self, text: str) -> Optional[Dict[str, Any]]:
        """Normalize decade expressions."""
        try:
            # Extract decade information
            decade_match = re.search(r'(\d{2,4})s', text)
            if decade_match:
                year = int(decade_match.group(1))
                if year < 100:
                    year = year + 1900 if year >= 20 else year + 2000
                
                # Handle early/mid/late modifiers
                if 'early' in text.lower():
                    return {"text": text, "start": year, "end": year + 2, "precision": "early_decade", "method": "spacy_decade"}
                elif 'mid' in text.lower():
                    return {"text": text, "start": year + 3, "end": year + 6, "precision": "mid_decade", "method": "spacy_decade"}
                elif 'late' in text.lower():
                    return {"text": text, "start": year + 7, "end": year + 9, "precision": "late_decade", "method": "spacy_decade"}
                else:
                    return {"text": text, "start": year, "end": year + 9, "precision": "decade", "method": "spacy_decade"}
            
            return None
        except Exception:
            return None
    
    def _normalize_relative_expression(self, text: str) -> Optional[Dict[str, Any]]:
        """Normalize relative expressions."""
        try:
            current_year = self.current_year
            
            if 'last decade' in text.lower() or 'past decade' in text.lower():
                return {"text": text, "start": current_year - 10, "end": current_year - 1, "precision": "decade", "method": "spacy_relative"}
            elif 'recent years' in text.lower():
                return {"text": text, "start": current_year - 5, "end": current_year, "precision": "recent", "method": "spacy_relative"}
            elif 'this decade' in text.lower():
                decade_start = current_year // 10 * 10
                return {"text": text, "start": decade_start, "end": current_year, "precision": "current_decade", "method": "spacy_relative"}
            
            # Handle "X years ago"
            years_match = re.search(r'(\d+)\s+years?\s+ago', text)
            if years_match:
                years_ago = int(years_match.group(1))
                target_year = current_year - years_ago
                return {"text": text, "start": target_year, "end": target_year, "precision": "year", "method": "spacy_relative"}
            
            return None
        except Exception:
            return None
    
    def _normalize_season_expression(self, text: str) -> Optional[Dict[str, Any]]:
        """Normalize season expressions."""
        try:
            season_match = re.search(r'(spring|summer|fall|autumn|winter).*?(\d{4})', text, re.IGNORECASE)
            if season_match:
                season = season_match.group(1).lower()
                year = int(season_match.group(2))
                
                season_months = {
                    "spring": (3, 5),
                    "summer": (6, 8),
                    "fall": (9, 11),
                    "autumn": (9, 11),
                    "winter": (12, 2)
                }
                
                start_month, end_month = season_months.get(season, (1, 12))
                
                return {
                    "text": text,
                    "start": f"{year}-{start_month:02d}-01",
                    "end": f"{year}-{end_month:02d}-28",
                    "precision": "season",
                    "season": season,
                    "year": year,
                    "method": "spacy_season"
                }
            
            return None
        except Exception:
            return None
    
    def _merge_and_deduplicate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from different methods and remove duplicates."""
        # Simple deduplication based on text similarity
        seen_texts = set()
        unique_expressions = []
        unique_normalized = []
        
        for expr in results["expressions"]:
            text_key = expr["text"].lower().strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_expressions.append(expr)
        
        # Process normalization with better logic
        for norm in results["normalized"]:
            if norm:  # Skip None results
                text_key = norm["text"].lower().strip()
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    unique_normalized.append(norm)
                else:
                    # Update existing with higher confidence
                    for i, existing in enumerate(unique_normalized):
                        if existing["text"].lower().strip() == text_key:
                            if norm.get("confidence", 0) > existing.get("confidence", 0):
                                unique_normalized[i] = norm
                            break
        
        # CRITICAL FIX: Force normalization of detected expressions that weren't normalized
        for expr in unique_expressions:
            expr_text = expr["text"].lower().strip()
            # Check if this expression was normalized
            already_normalized = any(norm["text"].lower().strip() == expr_text for norm in unique_normalized)
            
            if not already_normalized:
                # Force normalization using fallback method
                fallback_norm = self._fallback_normalize_expression(expr["text"], expr.get("method", "unknown"))
                if fallback_norm:
                    unique_normalized.append(fallback_norm)
        
        return {
            "expressions": unique_expressions,
            "normalized": unique_normalized,
            "analysis_methods": results["analysis_methods"],
            "total_expressions": len(unique_expressions)
        }
    
    def _semantic_temporal_analysis(self, text: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add semantic understanding to temporal analysis."""
        analysis = {
            "temporal_context": {},
            "confidence_level": "medium",
            "temporal_scope": "unknown"
        }
        
        try:
            # Analyze temporal scope
            if any(norm.get("precision") == "decade" for norm in results["normalized"]):
                analysis["temporal_scope"] = "decade"
            elif any(norm.get("precision") == "year" for norm in results["normalized"]):
                analysis["temporal_scope"] = "year"
            elif any(norm.get("precision") == "season" for norm in results["normalized"]):
                analysis["temporal_scope"] = "season"
            
            # Analyze confidence
            confidences = [expr.get("confidence", 0.5) for expr in results["expressions"]]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence >= 0.8:
                    analysis["confidence_level"] = "high"
                elif avg_confidence >= 0.6:
                    analysis["confidence_level"] = "medium"
                else:
                    analysis["confidence_level"] = "low"
            
            # Add temporal context
            analysis["temporal_context"] = {
                "has_relative_dates": any("relative" in expr.get("method", "") for expr in results["expressions"]),
                "has_absolute_dates": any("decade" in expr.get("method", "") or "year" in expr.get("method", "") for expr in results["expressions"]),
                "method_diversity": len(set(results["analysis_methods"])),
                "processing_sophisticated": len(results["analysis_methods"]) > 1
            }
            
        except Exception as e:
            self.logger.debug(f"Error in semantic analysis: {e}")
        
        return analysis
    
    def _fallback_normalize_expression(self, text: str, method: str) -> Optional[Dict[str, Any]]:
        """Fallback normalization for expressions that weren't normalized by specific methods."""
        try:
            text_lower = text.lower().strip()
            import re
            
            # Handle decade patterns (including complex ones)
            if 's' in text_lower and any(char.isdigit() for char in text_lower):
                # Handle compound decades: "80s and 90s", "1980s and 1990s"
                compound_match = re.search(r'(\d{2,4})s\s+(?:and|&)\s+(\d{2,4})s', text_lower)
                if compound_match:
                    year1 = int(compound_match.group(1))
                    year2 = int(compound_match.group(2))
                    
                    # Convert 2-digit to 4-digit years
                    if year1 < 100:
                        year1 = year1 + 1900 if year1 >= 20 else year1 + 2000
                    if year2 < 100:
                        year2 = year2 + 1900 if year2 >= 20 else year2 + 2000
                    
                    return {
                        "text": text,
                        "start": min(year1, year2),
                        "end": max(year1, year2) + 9,
                        "precision": "multi_decade",
                        "method": f"fallback_{method}"
                    }
                
                # Handle single decades with modifiers: "early 2000s", "late 90s"
                modifier_match = re.search(r'(early|mid|late)\s+(\d{2,4})s', text_lower)
                if modifier_match:
                    modifier = modifier_match.group(1)
                    year_part = int(modifier_match.group(2))
                    
                    if year_part < 100:
                        start_year = year_part + 1900 if year_part >= 20 else year_part + 2000
                    else:
                        start_year = year_part
                    
                    if modifier == "early":
                        return {"text": text, "start": start_year, "end": start_year + 2, "precision": "early_decade", "method": f"fallback_{method}"}
                    elif modifier == "mid":
                        return {"text": text, "start": start_year + 3, "end": start_year + 6, "precision": "mid_decade", "method": f"fallback_{method}"}
                    elif modifier == "late":
                        return {"text": text, "start": start_year + 7, "end": start_year + 9, "precision": "late_decade", "method": f"fallback_{method}"}
                
                # Handle standard decades
                decade_match = re.search(r'(\d{2,4})s', text_lower)
                if decade_match:
                    year_part = int(decade_match.group(1))
                    if year_part < 100:
                        # Two digit year: 90s -> 1990s or 00s -> 2000s  
                        start_year = year_part + 1900 if year_part >= 20 else year_part + 2000
                    else:
                        # Four digit year: 1990s -> 1990s
                        start_year = year_part
                    
                    return {
                        "text": text,
                        "start": start_year,
                        "end": start_year + 9,
                        "precision": "decade",
                        "method": f"fallback_{method}"
                    }
            
            # Handle "around X" patterns
            around_match = re.search(r'(?:around|about|circa)\s+(\d{4})', text_lower)
            if around_match:
                year = int(around_match.group(1))
                return {
                    "text": text,
                    "start": year - 2,
                    "end": year + 2,
                    "precision": "approximate_year",
                    "method": f"fallback_{method}"
                }
            
            # Handle year ranges: "between 1990 and 2000"
            range_match = re.search(r'(?:between\s+)?(\d{4})(?:\s+(?:and|to|-)\s+)(\d{4})', text_lower)
            if range_match:
                start_year = int(range_match.group(1))
                end_year = int(range_match.group(2))
                return {
                    "text": text,
                    "start": start_year,
                    "end": end_year,
                    "precision": "year_range",
                    "method": f"fallback_{method}"
                }
            
            # Handle "turn of century/millennium"
            if "turn" in text_lower and ("century" in text_lower or "millennium" in text_lower):
                if "millennium" in text_lower:
                    return {
                        "text": text,
                        "start": 1995,
                        "end": 2005,
                        "precision": "millennium_turn",
                        "method": f"fallback_{method}"
                    }
                else:  # century
                    return {
                        "text": text,
                        "start": 1995,
                        "end": 2005,
                        "precision": "century_turn",
                        "method": f"fallback_{method}"
                    }
            
            # Handle "post-X" patterns
            post_match = re.search(r'post-?(\w+)', text_lower)
            if post_match:
                term = post_match.group(1)
                if term in ["millennium", "2000"]:
                    return {
                        "text": text,
                        "start": 2000,
                        "end": self.current_year,
                        "precision": "post_period",
                        "method": f"fallback_{method}"
                    }
            
            # Handle relative patterns
            if "last decade" in text_lower or "past decade" in text_lower:
                current_year = self.current_year
                return {
                    "text": text,
                    "start": current_year - 10,
                    "end": current_year - 1,
                    "precision": "decade",
                    "method": f"fallback_{method}"
                }
            
            # Handle "X decades ago"
            if "decades ago" in text_lower:
                import re
                num_match = re.search(r'(\w+)\s+decades?\s+ago', text_lower)
                if num_match:
                    num_word = num_match.group(1)
                    # Convert word to number
                    word_to_num = {
                        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                        "a": 1, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5
                    }
                    decades_back = word_to_num.get(num_word, 2)  # Default to 2
                    target_year = self.current_year - (decades_back * 10)
                    
                    return {
                        "text": text,
                        "start": target_year,
                        "end": target_year + 9,
                        "precision": "decade",
                        "method": f"fallback_{method}"
                    }
            
            # Handle "this decade"
            if "this decade" in text_lower:
                decade_start = self.current_year // 10 * 10
                return {
                    "text": text,
                    "start": decade_start,
                    "end": self.current_year,
                    "precision": "current_decade",
                    "method": f"fallback_{method}"
                }
            
            # Handle single years
            if text_lower.isdigit() and len(text_lower) == 4:
                year = int(text_lower)
                if 1900 <= year <= 2030:
                    return {
                        "text": text,
                        "start": year,
                        "end": year,
                        "precision": "year",
                        "method": f"fallback_{method}"
                    }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error in fallback normalization: {e}")
            return None
    
    def _create_ingestion_search_tags(self, analysis: Dict[str, Any]) -> List[str]:
        """Create searchable tags for content ingestion."""
        tags = []
        
        # Add decade tags
        for norm in analysis.get("normalized", []):
            precision = norm.get("precision")
            start = norm.get("start")
            end = norm.get("end")
            
            if precision == "decade" and isinstance(start, int):
                decade = f"{start}s"
                tags.append(f"spacy_decade_{decade}")
            elif precision == "year" and isinstance(start, int):
                tags.append(f"spacy_year_{start}")
            elif precision == "current_decade":
                current_decade = (self.current_year // 10) * 10
                tags.append(f"spacy_decade_{current_decade}s")
        
        # Add method tags
        for method in analysis.get("analysis_methods", []):
            tags.append(f"spacy_method_{method}")
        
        return list(set(tags))  # Remove duplicates


# Maintain backward compatibility
SophisticatedTemporalPlugin = SpacyWithFallbackIngestionAndQueryPlugin
TemporalExpressionPlugin = SpacyWithFallbackIngestionAndQueryPlugin