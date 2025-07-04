"""
Gensim-based similarity engine for movie search.
Handles TF-IDF vectorization, LSI/LDA topic modeling, and similarity calculations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import pickle
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from gensim import corpora, models, similarities
    from gensim.models import TfidfModel, LsiModel, LdaModel
    from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .text_processor import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of similarity calculation."""
    document_id: str
    similarity_score: float
    matched_terms: List[str]
    field_scores: Dict[str, float]


@dataclass
class ModelStats:
    """Statistics for similarity models."""
    num_documents: int
    num_terms: int
    last_updated: datetime
    model_type: str
    cache_size: int


class SimilarityEngine:
    """Gensim-based similarity engine for movie search."""
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 min_similarity: float = 0.1,
                 max_cache_size: int = 10000):
        """Initialize similarity engine.
        
        Args:
            cache_dir: Directory for caching models
            min_similarity: Minimum similarity threshold
            max_cache_size: Maximum number of cached similarity matrices
        """
        self.cache_dir = cache_dir or Path("cache/similarity")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_similarity = min_similarity
        self.max_cache_size = max_cache_size
        
        # Text processor for preprocessing
        self.text_processor = TextProcessor()
        
        # Gensim models
        self.dictionary: Optional[Dictionary] = None
        self.tfidf_model: Optional[TfidfModel] = None
        self.lsi_model: Optional[LsiModel] = None
        self.lda_model: Optional[LdaModel] = None
        
        # Similarity indices
        self.tfidf_index: Optional[SparseMatrixSimilarity] = None
        self.lsi_index: Optional[MatrixSimilarity] = None
        self.lda_index: Optional[MatrixSimilarity] = None
        
        # Sklearn fallback
        self.sklearn_tfidf: Optional[TfidfVectorizer] = None
        self.sklearn_vectors: Optional[np.ndarray] = None
        
        # Document corpus
        self.document_corpus: List[List[str]] = []
        self.document_ids: List[str] = []
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Model statistics
        self.stats = ModelStats(
            num_documents=0,
            num_terms=0,
            last_updated=datetime.now(),
            model_type="none",
            cache_size=0
        )
        
        # Initialize if libraries are available
        if not GENSIM_AVAILABLE and not SKLEARN_AVAILABLE:
            logger.warning("Neither Gensim nor Scikit-learn available - similarity search disabled")
    
    def _ensure_gensim_available(self) -> bool:
        """Check if Gensim is available."""
        if not GENSIM_AVAILABLE:
            logger.warning("Gensim not available - falling back to sklearn")
            return False
        return True
    
    def add_document(self, doc_id: str, content: Dict[str, Any]) -> None:
        """Add a document to the corpus.
        
        Args:
            doc_id: Unique document identifier
            content: Document content with field mappings
        """
        # Process all text fields
        processed_terms = []
        field_terms = {}
        
        for field_name, field_value in content.items():
            if field_value:
                terms = self.text_processor.process_movie_field(field_name, field_value)
                if terms:
                    processed_terms.extend(terms)
                    field_terms[field_name] = terms
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in processed_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        # Store document
        if doc_id in self.document_ids:
            # Update existing document
            idx = self.document_ids.index(doc_id)
            self.document_corpus[idx] = unique_terms
            self.document_metadata[doc_id] = {
                "field_terms": field_terms,
                "total_terms": len(unique_terms),
                "updated_at": datetime.now()
            }
        else:
            # Add new document
            self.document_ids.append(doc_id)
            self.document_corpus.append(unique_terms)
            self.document_metadata[doc_id] = {
                "field_terms": field_terms,
                "total_terms": len(unique_terms),
                "added_at": datetime.now()
            }
        
        logger.debug(f"Added document {doc_id} with {len(unique_terms)} terms")
    
    def build_models(self, 
                    use_tfidf: bool = True,
                    use_lsi: bool = True,
                    use_lda: bool = False,
                    lsi_topics: int = 100,
                    lda_topics: int = 50) -> bool:
        """Build similarity models from the corpus.
        
        Args:
            use_tfidf: Build TF-IDF model
            use_lsi: Build LSI model
            use_lda: Build LDA model
            lsi_topics: Number of LSI topics
            lda_topics: Number of LDA topics
            
        Returns:
            True if models built successfully
        """
        if not self.document_corpus:
            logger.warning("No documents in corpus - cannot build models")
            return False
        
        logger.info(f"Building similarity models for {len(self.document_corpus)} documents")
        
        try:
            if GENSIM_AVAILABLE:
                return self._build_gensim_models(use_tfidf, use_lsi, use_lda, lsi_topics, lda_topics)
            elif SKLEARN_AVAILABLE:
                return self._build_sklearn_models()
            else:
                logger.error("No similarity libraries available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to build models: {e}")
            return False
    
    def _build_gensim_models(self,
                            use_tfidf: bool,
                            use_lsi: bool,
                            use_lda: bool,
                            lsi_topics: int,
                            lda_topics: int) -> bool:
        """Build Gensim-based models."""
        
        # Create dictionary
        self.dictionary = Dictionary(self.document_corpus)
        
        # Filter extremely rare and common words
        self.dictionary.filter_extremes(
            no_below=2,  # Must appear in at least 2 documents
            no_above=0.8,  # Must appear in less than 80% of documents
            keep_n=10000  # Keep most frequent 10000 words
        )
        
        # Create bag-of-words corpus
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.document_corpus]
        
        # Build TF-IDF model
        if use_tfidf:
            self.tfidf_model = TfidfModel(bow_corpus)
            tfidf_corpus = self.tfidf_model[bow_corpus]
            
            # Build TF-IDF similarity index
            self.tfidf_index = SparseMatrixSimilarity(
                tfidf_corpus,
                num_features=len(self.dictionary)
            )
            
            logger.info("Built TF-IDF model and index")
        
        # Build LSI model
        if use_lsi and self.tfidf_model:
            tfidf_corpus = self.tfidf_model[bow_corpus]
            self.lsi_model = LsiModel(
                tfidf_corpus,
                id2word=self.dictionary,
                num_topics=lsi_topics
            )
            
            # Build LSI similarity index
            lsi_corpus = self.lsi_model[tfidf_corpus]
            self.lsi_index = MatrixSimilarity(
                lsi_corpus,
                num_features=lsi_topics
            )
            
            logger.info(f"Built LSI model with {lsi_topics} topics")
        
        # Build LDA model
        if use_lda:
            self.lda_model = LdaModel(
                bow_corpus,
                id2word=self.dictionary,
                num_topics=lda_topics,
                random_state=42,
                passes=3
            )
            
            # Build LDA similarity index
            lda_corpus = self.lda_model[bow_corpus]
            self.lda_index = MatrixSimilarity(
                lda_corpus,
                num_features=lda_topics
            )
            
            logger.info(f"Built LDA model with {lda_topics} topics")
        
        # Update stats
        self.stats = ModelStats(
            num_documents=len(self.document_corpus),
            num_terms=len(self.dictionary),
            last_updated=datetime.now(),
            model_type="gensim",
            cache_size=len(self.document_metadata)
        )
        
        return True
    
    def _build_sklearn_models(self) -> bool:
        """Build scikit-learn based models as fallback."""
        
        # Convert documents to text
        documents = []
        for doc_terms in self.document_corpus:
            documents.append(" ".join(doc_terms))
        
        # Build TF-IDF vectorizer
        self.sklearn_tfidf = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        # Fit and transform documents
        self.sklearn_vectors = self.sklearn_tfidf.fit_transform(documents)
        
        # Update stats
        self.stats = ModelStats(
            num_documents=len(documents),
            num_terms=len(self.sklearn_tfidf.get_feature_names_out()),
            last_updated=datetime.now(),
            model_type="sklearn",
            cache_size=len(self.document_metadata)
        )
        
        logger.info("Built scikit-learn TF-IDF model")
        return True
    
    def find_similar(self,
                    query: str,
                    top_k: int = 10,
                    model_type: str = "tfidf") -> List[SimilarityResult]:
        """Find similar documents for a query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            model_type: Type of model to use ("tfidf", "lsi", "lda")
            
        Returns:
            List of similarity results
        """
        if not self.document_corpus:
            logger.warning("No documents in corpus")
            return []
        
        # Process query
        processed_query = self.text_processor.process_text(query)
        query_terms = processed_query.keywords
        
        if not query_terms:
            logger.warning("Query produced no valid terms")
            return []
        
        try:
            if self.stats.model_type == "gensim":
                return self._find_similar_gensim(query_terms, top_k, model_type)
            elif self.stats.model_type == "sklearn":
                return self._find_similar_sklearn(query_terms, top_k)
            else:
                logger.error("No models available for similarity search")
                return []
                
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def _find_similar_gensim(self,
                           query_terms: List[str],
                           top_k: int,
                           model_type: str) -> List[SimilarityResult]:
        """Find similar documents using Gensim models."""
        
        if not self.dictionary:
            logger.error("Dictionary not available")
            return []
        
        # Convert query to bag of words
        query_bow = self.dictionary.doc2bow(query_terms)
        
        if model_type == "tfidf" and self.tfidf_model and self.tfidf_index:
            # TF-IDF similarity
            query_tfidf = self.tfidf_model[query_bow]
            similarities = self.tfidf_index[query_tfidf]
            
        elif model_type == "lsi" and self.lsi_model and self.lsi_index:
            # LSI similarity
            query_tfidf = self.tfidf_model[query_bow] if self.tfidf_model else query_bow
            query_lsi = self.lsi_model[query_tfidf]
            similarities = self.lsi_index[query_lsi]
            
        elif model_type == "lda" and self.lda_model and self.lda_index:
            # LDA similarity
            query_lda = self.lda_model[query_bow]
            similarities = self.lda_index[query_lda]
            
        else:
            logger.error(f"Model type {model_type} not available")
            return []
        
        # Get top results
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= self.min_similarity:
                doc_id = self.document_ids[i]
                
                # Calculate matched terms
                doc_terms = set(self.document_corpus[i])
                query_terms_set = set(query_terms)
                matched_terms = list(doc_terms.intersection(query_terms_set))
                
                # Calculate field scores
                field_scores = self._calculate_field_scores(doc_id, query_terms)
                
                results.append(SimilarityResult(
                    document_id=doc_id,
                    similarity_score=float(similarity),
                    matched_terms=matched_terms,
                    field_scores=field_scores
                ))
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:top_k]
    
    def _find_similar_sklearn(self,
                            query_terms: List[str],
                            top_k: int) -> List[SimilarityResult]:
        """Find similar documents using scikit-learn models."""
        
        if not self.sklearn_tfidf or self.sklearn_vectors is None:
            logger.error("Scikit-learn models not available")
            return []
        
        # Convert query to text
        query_text = " ".join(query_terms)
        
        # Transform query
        query_vector = self.sklearn_tfidf.transform([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.sklearn_vectors)[0]
        
        # Get top results
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= self.min_similarity:
                doc_id = self.document_ids[i]
                
                # Calculate matched terms
                doc_terms = set(self.document_corpus[i])
                query_terms_set = set(query_terms)
                matched_terms = list(doc_terms.intersection(query_terms_set))
                
                # Calculate field scores
                field_scores = self._calculate_field_scores(doc_id, query_terms)
                
                results.append(SimilarityResult(
                    document_id=doc_id,
                    similarity_score=float(similarity),
                    matched_terms=matched_terms,
                    field_scores=field_scores
                ))
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:top_k]
    
    def _calculate_field_scores(self,
                              doc_id: str,
                              query_terms: List[str]) -> Dict[str, float]:
        """Calculate field-specific similarity scores."""
        field_scores = {}
        
        if doc_id not in self.document_metadata:
            return field_scores
        
        field_terms = self.document_metadata[doc_id].get("field_terms", {})
        query_terms_set = set(query_terms)
        
        for field_name, terms in field_terms.items():
            if terms:
                field_terms_set = set(terms)
                intersection = field_terms_set.intersection(query_terms_set)
                
                if intersection:
                    # Calculate Jaccard similarity
                    union = field_terms_set.union(query_terms_set)
                    score = len(intersection) / len(union)
                    field_scores[field_name] = score
        
        return field_scores
    
    def get_document_topics(self, doc_id: str, model_type: str = "lsi") -> Optional[List[Tuple[int, float]]]:
        """Get topics for a specific document.
        
        Args:
            doc_id: Document ID
            model_type: Model type ("lsi" or "lda")
            
        Returns:
            List of (topic_id, weight) tuples
        """
        if doc_id not in self.document_ids:
            return None
        
        doc_idx = self.document_ids.index(doc_id)
        doc_terms = self.document_corpus[doc_idx]
        
        if not self.dictionary:
            return None
        
        query_bow = self.dictionary.doc2bow(doc_terms)
        
        if model_type == "lsi" and self.lsi_model:
            query_tfidf = self.tfidf_model[query_bow] if self.tfidf_model else query_bow
            topics = self.lsi_model[query_tfidf]
            return topics
            
        elif model_type == "lda" and self.lda_model:
            topics = self.lda_model[query_bow]
            return topics
        
        return None
    
    def save_models(self, filepath: Optional[Path] = None) -> bool:
        """Save models to disk.
        
        Args:
            filepath: Path to save models
            
        Returns:
            True if saved successfully
        """
        if filepath is None:
            filepath = self.cache_dir / f"similarity_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        try:
            model_data = {
                "stats": self.stats,
                "document_ids": self.document_ids,
                "document_corpus": self.document_corpus,
                "document_metadata": self.document_metadata,
                "dictionary": self.dictionary,
                "tfidf_model": self.tfidf_model,
                "lsi_model": self.lsi_model,
                "lda_model": self.lda_model,
                "sklearn_tfidf": self.sklearn_tfidf,
                "sklearn_vectors": self.sklearn_vectors
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved similarity models to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False
    
    def load_models(self, filepath: Path) -> bool:
        """Load models from disk.
        
        Args:
            filepath: Path to load models from
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.stats = model_data.get("stats", self.stats)
            self.document_ids = model_data.get("document_ids", [])
            self.document_corpus = model_data.get("document_corpus", [])
            self.document_metadata = model_data.get("document_metadata", {})
            self.dictionary = model_data.get("dictionary")
            self.tfidf_model = model_data.get("tfidf_model")
            self.lsi_model = model_data.get("lsi_model")
            self.lda_model = model_data.get("lda_model")
            self.sklearn_tfidf = model_data.get("sklearn_tfidf")
            self.sklearn_vectors = model_data.get("sklearn_vectors")
            
            # Rebuild indices if needed
            if self.tfidf_model and self.document_corpus:
                self._rebuild_indices()
            
            logger.info(f"Loaded similarity models from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def _rebuild_indices(self) -> None:
        """Rebuild similarity indices after loading models."""
        if not self.dictionary or not self.document_corpus:
            return
        
        try:
            bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.document_corpus]
            
            if self.tfidf_model:
                tfidf_corpus = self.tfidf_model[bow_corpus]
                self.tfidf_index = SparseMatrixSimilarity(
                    tfidf_corpus,
                    num_features=len(self.dictionary)
                )
            
            if self.lsi_model:
                tfidf_corpus = self.tfidf_model[bow_corpus]
                lsi_corpus = self.lsi_model[tfidf_corpus]
                self.lsi_index = MatrixSimilarity(
                    lsi_corpus,
                    num_features=self.lsi_model.num_topics
                )
            
            if self.lda_model:
                lda_corpus = self.lda_model[bow_corpus]
                self.lda_index = MatrixSimilarity(
                    lda_corpus,
                    num_features=self.lda_model.num_topics
                )
                
            logger.debug("Rebuilt similarity indices")
            
        except Exception as e:
            logger.error(f"Failed to rebuild indices: {e}")
    
    def get_stats(self) -> ModelStats:
        """Get model statistics."""
        return self.stats
    
    def clear_cache(self) -> None:
        """Clear document cache."""
        self.document_corpus.clear()
        self.document_ids.clear()
        self.document_metadata.clear()
        
        self.stats = ModelStats(
            num_documents=0,
            num_terms=0,
            last_updated=datetime.now(),
            model_type="none",
            cache_size=0
        )
        
        logger.info("Cleared similarity cache")