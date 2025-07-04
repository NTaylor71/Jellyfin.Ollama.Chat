"""
Semantic role labeling plugin for extracting predicate-argument structures.
Identifies WHO did WHAT to WHOM relationships in text.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import logging

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .base import DualUsePlugin


class SemanticRoleLabelerPlugin(DualUsePlugin):
    """Extract semantic roles and predicate-argument structures."""
    
    def __init__(self):
        super().__init__()
        
    def _initialize_models(self):
        """Initialize spaCy model and semantic role patterns."""
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spaCy model for semantic role labeling")
            except OSError:
                self.logger.warning("spaCy model not found, using pattern matching fallback")
                self.nlp = None
        else:
            self.nlp = None
            self.logger.warning("spaCy not available, using pattern matching fallback")
        
        # Define common semantic patterns for media content
        self.action_patterns = [
            # Film/media production patterns
            (r'(\w+(?:\s+\w+)*)\s+(directed|produced|wrote|created|starred in|acted in|composed|scored)\s+(.+)', 
             lambda m: {"agent": m.group(1).strip(), "predicate": m.group(2), "theme": m.group(3).strip(), "frame": "Behind_the_scenes"}),
            
            # Performance patterns
            (r'(\w+(?:\s+\w+)*)\s+(plays|portrays|voices|performs)\s+(.+)',
             lambda m: {"agent": m.group(1).strip(), "predicate": m.group(2), "theme": m.group(3).strip(), "frame": "Performance"}),
            
            # Content description patterns
            (r'(.+)\s+(follows|tells the story of|is about|centers on|focuses on)\s+(.+)',
             lambda m: {"agent": m.group(1).strip(), "predicate": m.group(2), "theme": m.group(3).strip(), "frame": "Narrative"}),
            
            # Character action patterns
            (r'(\w+(?:\s+\w+)*)\s+(fights|battles|saves|rescues|discovers|finds|kills|defeats)\s+(.+)',
             lambda m: {"agent": m.group(1).strip(), "predicate": m.group(2), "theme": m.group(3).strip(), "frame": "Action"}),
            
            # Relationship patterns
            (r'(\w+(?:\s+\w+)*)\s+(loves|hates|meets|marries|befriends|betrays)\s+(.+)',
             lambda m: {"agent": m.group(1).strip(), "predicate": m.group(2), "theme": m.group(3).strip(), "frame": "Relationship"}),
            
            # Creation patterns
            (r'(\w+(?:\s+\w+)*)\s+(creates|builds|makes|invents|develops)\s+(.+)',
             lambda m: {"agent": m.group(1).strip(), "predicate": m.group(2), "theme": m.group(3).strip(), "frame": "Creating"}),
        ]
        
        # VerbNet classes for common media verbs
        self.verbnet_classes = {
            "directed": "29.8",  # Performance
            "produced": "26.4",  # Build
            "starred": "29.8",   # Performance
            "acted": "29.8",     # Performance
            "wrote": "25.2",     # Create
            "composed": "25.2",  # Create
            "plays": "29.8",     # Performance
            "portrays": "29.8",  # Performance
            "fights": "36.4",    # Combat
            "battles": "36.4",   # Combat
            "saves": "10.5",     # Rescue
            "rescues": "10.5",   # Rescue
            "loves": "31.2",     # Admire
            "hates": "31.2",     # Admire (negative)
            "meets": "36.3",     # Meet
            "follows": "51.6",   # Accompany
            "creates": "26.4",   # Build
            "builds": "26.4",    # Build
            "discovers": "30.2", # Learn
            "finds": "30.2",     # Learn
        }
        
        # FrameNet frames for media contexts
        self.framenet_frames = {
            "Behind_the_scenes": ["directing", "producing", "writing", "composing", "filming"],
            "Performance": ["acting", "starring", "playing", "portraying", "voicing"],
            "Narrative": ["story", "plot", "telling", "following", "about"],
            "Action": ["fighting", "battling", "saving", "rescuing", "defeating"],
            "Relationship": ["love", "friendship", "romance", "conflict", "betrayal"],
            "Creating": ["creation", "building", "making", "invention", "development"],
        }
    
    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract semantic roles and predicate-argument structures."""
        try:
            semantic_roles = []
            
            if self.nlp:
                # Use spaCy for more sophisticated analysis
                semantic_roles.extend(self._extract_with_spacy(text))
            else:
                # Fallback to pattern matching
                semantic_roles.extend(self._extract_with_patterns(text))
            
            # Post-process and clean results
            cleaned_roles = self._clean_semantic_roles(semantic_roles)
            
            # Group by frame types
            frames_by_type = self._group_by_frames(cleaned_roles)
            
            return {
                "semantic_roles": cleaned_roles,
                "frames_by_type": frames_by_type,
                "role_count": len(cleaned_roles),
                "frames_detected": list(frames_by_type.keys()),
                "main_predicates": [role.get("predicate", "") for role in cleaned_roles[:5]]
            }
            
        except Exception as e:
            self.logger.error(f"Error in semantic role analysis: {e}")
            return {"error": str(e), "semantic_roles": []}
    
    def _extract_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract semantic roles using spaCy dependency parsing."""
        roles = []
        
        try:
            doc = self.nlp(text)
            
            for sent in doc.sents:
                # Find main verbs (predicates)
                for token in sent:
                    if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp", "xcomp"]:
                        predicate = token.lemma_
                        
                        # Find arguments
                        agent = None
                        theme = None
                        
                        # Find subject (agent)
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                agent = self._extract_entity_phrase(child)
                        
                        # Find object (theme)
                        for child in token.children:
                            if child.dep_ in ["dobj", "pobj", "attr"]:
                                theme = self._extract_entity_phrase(child)
                        
                        # Find prepositional phrases that might be themes
                        if not theme:
                            for child in token.children:
                                if child.dep_ == "prep":
                                    for grandchild in child.children:
                                        if grandchild.dep_ == "pobj":
                                            theme = self._extract_entity_phrase(grandchild)
                                            break
                        
                        if agent and predicate:
                            frame = self._determine_frame(predicate)
                            verbnet_class = self.verbnet_classes.get(predicate.lower(), "unknown")
                            
                            roles.append({
                                "predicate": predicate,
                                "agent": agent,
                                "theme": theme or "unspecified",
                                "frame": frame,
                                "verbnet_class": verbnet_class,
                                "confidence": 0.8  # spaCy-based extraction has higher confidence
                            })
        
        except Exception as e:
            self.logger.debug(f"spaCy extraction error: {e}")
        
        return roles
    
    def _extract_entity_phrase(self, token) -> str:
        """Extract the full noun phrase around a token."""
        # Get the root and its subtree
        phrase_tokens = []
        
        # Add tokens to the left
        current = token
        while current.i > 0 and (current.dep_ in ["compound", "amod", "det"] or 
                                 current.pos_ in ["DET", "ADJ"] or
                                 (current.pos_ == "NOUN" and current.head == token)):
            current = current.doc[current.i - 1]
            if current.pos_ in ["NOUN", "PROPN", "ADJ", "DET"]:
                phrase_tokens.insert(0, current.text)
        
        # Add the main token
        phrase_tokens.append(token.text)
        
        # Add tokens to the right
        for child in token.children:
            if child.dep_ in ["compound", "amod"] or child.pos_ in ["NOUN", "PROPN"]:
                phrase_tokens.append(child.text)
        
        return " ".join(phrase_tokens).strip()
    
    def _extract_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract semantic roles using regex patterns."""
        roles = []
        
        for pattern, handler in self.action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    role_data = handler(match)
                    predicate = role_data["predicate"]
                    
                    # Add VerbNet class
                    role_data["verbnet_class"] = self.verbnet_classes.get(predicate.lower(), "unknown")
                    role_data["confidence"] = 0.6  # Pattern-based has lower confidence
                    
                    roles.append(role_data)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing pattern match: {e}")
        
        return roles
    
    def _determine_frame(self, predicate: str) -> str:
        """Determine the FrameNet frame for a predicate."""
        predicate_lower = predicate.lower()
        
        for frame, verbs in self.framenet_frames.items():
            if any(verb in predicate_lower for verb in verbs):
                return frame
        
        # Default frame based on predicate type
        if predicate_lower in ["direct", "produce", "write", "compose"]:
            return "Behind_the_scenes"
        elif predicate_lower in ["act", "star", "play", "portray"]:
            return "Performance"
        elif predicate_lower in ["fight", "battle", "save", "rescue"]:
            return "Action"
        elif predicate_lower in ["love", "hate", "meet", "marry"]:
            return "Relationship"
        elif predicate_lower in ["create", "build", "make", "invent"]:
            return "Creating"
        else:
            return "General"
    
    def _clean_semantic_roles(self, roles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and filter semantic roles."""
        cleaned = []
        
        for role in roles:
            # Clean up text fields
            agent = self._clean_entity_text(role.get("agent", ""))
            theme = self._clean_entity_text(role.get("theme", ""))
            predicate = role.get("predicate", "").strip().lower()
            
            # Filter out low-quality extractions
            if (len(agent) > 1 and len(predicate) > 1 and 
                not any(word in agent.lower() for word in ["the", "a", "an", "this", "that"]) and
                not any(word in theme.lower() for word in ["the", "a", "an", "this", "that"]) if theme != "unspecified" else True):
                
                cleaned.append({
                    "predicate": predicate,
                    "agent": agent,
                    "theme": theme,
                    "frame": role.get("frame", "General"),
                    "verbnet_class": role.get("verbnet_class", "unknown"),
                    "confidence": role.get("confidence", 0.5)
                })
        
        # Remove duplicates
        unique_roles = []
        seen = set()
        for role in cleaned:
            key = (role["agent"], role["predicate"], role["theme"])
            if key not in seen:
                seen.add(key)
                unique_roles.append(role)
        
        return unique_roles[:10]  # Limit results
    
    def _clean_entity_text(self, text: str) -> str:
        """Clean entity text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading articles
        cleaned = re.sub(r'^(the|a|an)\s+', '', cleaned, flags=re.IGNORECASE)
        
        # Capitalize properly
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        return cleaned
    
    def _group_by_frames(self, roles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group semantic roles by FrameNet frames."""
        frames = {}
        
        for role in roles:
            frame = role.get("frame", "General")
            if frame not in frames:
                frames[frame] = []
            frames[frame].append(role)
        
        return frames