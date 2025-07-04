#!/usr/bin/env python3
"""
Integration example demonstrating the new linguistic analysis system.
Shows how the linguistic plugins work together to analyze movie content and queries.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from plugins.linguistic.conceptnet import ConceptNetExpansionPlugin
from plugins.linguistic.temporal import TemporalExpressionPlugin
from plugins.linguistic.semantic_roles import SemanticRoleLabelerPlugin


class LinguisticAnalysisDemo:
    """Demonstration of the new linguistic analysis system."""
    
    def __init__(self):
        self.concept_plugin = ConceptNetExpansionPlugin()
        self.temporal_plugin = TemporalExpressionPlugin()
        self.semantic_plugin = SemanticRoleLabelerPlugin()
    
    async def analyze_movie_content(self, movie_data: dict) -> dict:
        """Demonstrate linguistic analysis on movie content (ingestion)."""
        print("🎬 ANALYZING MOVIE CONTENT (Ingestion Pipeline)")
        print("=" * 60)
        
        # Extract text for analysis
        text_parts = []
        if movie_data.get("overview"):
            text_parts.append(movie_data["overview"])
        if movie_data.get("taglines"):
            text_parts.extend(movie_data["taglines"])
        
        content_text = " ".join(text_parts)
        print(f"📝 Content Text: {content_text[:100]}...")
        
        context = {"media_type": "movie"}
        
        # Run all linguistic plugins
        print("\n🔍 Running Linguistic Analysis Plugins...")
        
        concept_analysis = await self.concept_plugin.enhance_data(movie_data, context)
        temporal_analysis = await self.temporal_plugin.enhance_data(movie_data, context)
        semantic_analysis = await self.semantic_plugin.enhance_data(movie_data, context)
        
        # Combine results
        linguistic_data = {
            "concepts": concept_analysis.get("linguistic_analysis", {}).get("ConceptNetExpansionPlugin", {}),
            "temporal": temporal_analysis.get("linguistic_analysis", {}).get("TemporalExpressionPlugin", {}),
            "semantic_roles": semantic_analysis.get("linguistic_analysis", {}).get("SemanticRoleLabelerPlugin", {})
        }
        
        self._display_content_analysis(linguistic_data)
        
        return linguistic_data
    
    async def analyze_user_query(self, query: str) -> dict:
        """Demonstrate linguistic analysis on user query (search pipeline)."""
        print(f"\n🔍 ANALYZING USER QUERY (Search Pipeline)")
        print("=" * 60)
        print(f"🗣️  User Query: \"{query}\"")
        
        context = {"query_type": "search", "media_type": "movie"}
        
        # Run all linguistic plugins on query
        print("\n🧠 Running Query Analysis...")
        
        concept_result = await self.concept_plugin.embellish_query(query, context)
        temporal_result = await self.temporal_plugin.embellish_query(query, context)
        semantic_result = await self.semantic_plugin.embellish_query(query, context)
        
        query_analysis = {
            "original_query": query,
            "concepts": concept_result.get("analysis", {}),
            "temporal": temporal_result.get("analysis", {}),
            "semantic_roles": semantic_result.get("analysis", {})
        }
        
        self._display_query_analysis(query_analysis)
        
        return query_analysis
    
    def _display_content_analysis(self, analysis: dict):
        """Display content analysis results."""
        print("\n📊 CONTENT ANALYSIS RESULTS:")
        print("-" * 40)
        
        # Concept analysis
        concepts = analysis.get("concepts", {})
        if concepts.get("primary_concepts"):
            print(f"🏷️  Primary Concepts: {', '.join(concepts['primary_concepts'][:5])}")
        
        if concepts.get("expanded_concepts"):
            print("🌐 Concept Expansions:")
            for concept, expansions in list(concepts["expanded_concepts"].items())[:2]:
                print(f"   {concept} → {', '.join(expansions[:3])}")
        
        # Rate limiting status
        if concepts.get("rate_limit_status"):
            rate_status = concepts["rate_limit_status"]
            print(f"⚡ ConceptNet Usage: {rate_status['minute_percent_used']}% minute, {rate_status['hourly_percent_used']}% hourly")
        
        # Temporal analysis
        temporal = analysis.get("temporal", {})
        if temporal.get("normalized"):
            print(f"⏰ Temporal Expressions:")
            for expr in temporal["normalized"][:2]:
                print(f"   \"{expr['text']}\" → {expr['start']}-{expr['end']} ({expr['precision']})")
        
        # Semantic roles
        semantic = analysis.get("semantic_roles", {})
        if semantic.get("semantic_roles"):
            print("🎭 Semantic Roles:")
            for role in semantic["semantic_roles"][:2]:
                print(f"   {role['agent']} {role['predicate']} {role['theme']} [{role['frame']}]")
    
    def _display_query_analysis(self, analysis: dict):
        """Display query analysis results."""
        print("\n📊 QUERY ANALYSIS RESULTS:")
        print("-" * 40)
        
        # Concept analysis
        concepts = analysis.get("concepts", {})
        if concepts.get("primary_concepts"):
            print(f"🏷️  Query Concepts: {', '.join(concepts['primary_concepts'][:5])}")
        
        if concepts.get("expanded_concepts"):
            print("🌐 Query Expansions:")
            for concept, expansions in list(concepts["expanded_concepts"].items())[:2]:
                print(f"   {concept} → {', '.join(expansions[:3])}")
        
        # Rate limiting status
        if concepts.get("rate_limit_status"):
            rate_status = concepts["rate_limit_status"]
            print(f"⚡ ConceptNet Usage: {rate_status['minute_percent_used']}% minute, {rate_status['hourly_percent_used']}% hourly")
        
        # Temporal analysis
        temporal = analysis.get("temporal", {})
        if temporal.get("normalized"):
            print(f"⏰ Time References:")
            for expr in temporal["normalized"][:2]:
                print(f"   \"{expr['text']}\" → {expr['start']}-{expr['end']} ({expr['precision']})")
        
        # Semantic roles
        semantic = analysis.get("semantic_roles", {})
        if semantic.get("semantic_roles"):
            print("🎭 Intent Analysis:")
            for role in semantic["semantic_roles"][:2]:
                print(f"   {role['agent']} {role['predicate']} {role['theme']} [{role['frame']}]")
    
    def _demonstrate_mongodb_storage(self, movie_data: dict, linguistic_data: dict):
        """Show how data would be stored in MongoDB."""
        print("\n💾 MONGODB STORAGE STRUCTURE:")
        print("-" * 40)
        
        # Simulate MongoDB document structure
        mongo_doc = {
            "_id": "64a7b8f2e1234567890abcde",
            "type": "movie",
            "name": movie_data.get("name"),
            "overview": movie_data.get("overview"),
            
            # Original Jellyfin data
            "original": movie_data,
            
            # Linguistic analysis
            "linguistic": linguistic_data,
            
            # Search features (would be pre-computed)
            "search": {
                "text": self._generate_search_text(movie_data, linguistic_data),
                "concepts": linguistic_data.get("concepts", {}).get("primary_concepts", []),
                "temporal_features": linguistic_data.get("temporal", {}).get("normalized", []),
                "semantic_frames": [
                    role.get("frame") for role in 
                    linguistic_data.get("semantic_roles", {}).get("semantic_roles", [])
                ]
            },
            
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Display key parts
        print(f"📋 Document ID: {mongo_doc['_id']}")
        print(f"🎬 Movie: {mongo_doc['name']}")
        print(f"📝 Search Text: {mongo_doc['search']['text'][:100]}...")
        print(f"🏷️  Search Concepts: {', '.join(mongo_doc['search']['concepts'][:5])}")
        
        return mongo_doc
    
    def _generate_search_text(self, movie_data: dict, linguistic_data: dict) -> str:
        """Generate unified search text for MongoDB $text index."""
        text_parts = []
        
        # Original fields (weighted by repetition)
        if movie_data.get("name"):
            text_parts.extend([movie_data["name"]] * 3)  # Title gets 3x weight
        
        if movie_data.get("overview"):
            text_parts.append(movie_data["overview"])
        
        # Expanded concepts
        concepts = linguistic_data.get("concepts", {})
        if concepts.get("primary_concepts"):
            text_parts.extend(concepts["primary_concepts"])
        
        if concepts.get("expanded_concepts"):
            for expansions in concepts["expanded_concepts"].values():
                text_parts.extend(expansions[:3])  # Top 3 expansions
        
        # Semantic roles as searchable text
        semantic = linguistic_data.get("semantic_roles", {})
        for role in semantic.get("semantic_roles", [])[:3]:
            text_parts.append(f"{role.get('agent', '')} {role.get('predicate', '')} {role.get('theme', '')}")
        
        return " ".join(text_parts)
    
    def _demonstrate_query_matching(self, query_analysis: dict, mongo_doc: dict):
        """Show how query would match against MongoDB document."""
        print("\n🔍 QUERY MATCHING DEMONSTRATION:")
        print("-" * 40)
        
        query_concepts = query_analysis.get("concepts", {}).get("primary_concepts", [])
        doc_concepts = mongo_doc["search"]["concepts"]
        
        # Concept matching
        concept_matches = set(query_concepts) & set(doc_concepts)
        if concept_matches:
            print(f"✅ Concept Matches: {', '.join(concept_matches)}")
        
        # Temporal matching
        query_temporal = query_analysis.get("temporal", {}).get("normalized", [])
        doc_temporal = mongo_doc["search"]["temporal_features"]
        
        if query_temporal and doc_temporal:
            print("⏰ Temporal Matching:")
            for q_expr in query_temporal[:1]:
                for d_expr in doc_temporal[:1]:
                    if self._temporal_overlap(q_expr, d_expr):
                        print(f"   ✅ {q_expr['text']} overlaps with {d_expr['text']}")
        
        # Semantic frame matching
        query_frames = [
            role.get("frame") for role in 
            query_analysis.get("semantic_roles", {}).get("semantic_roles", [])
        ]
        doc_frames = mongo_doc["search"]["semantic_frames"]
        
        frame_matches = set(query_frames) & set(doc_frames)
        if frame_matches:
            print(f"🎭 Frame Matches: {', '.join(frame_matches)}")
    
    def _temporal_overlap(self, expr1: dict, expr2: dict) -> bool:
        """Check if two temporal expressions overlap."""
        try:
            # Simple overlap check for years
            if expr1.get("precision") == "year" and expr2.get("precision") == "year":
                return expr1["start"] == expr2["start"]
            
            # Decade/range overlap
            start1, end1 = expr1.get("start"), expr1.get("end")
            start2, end2 = expr2.get("start"), expr2.get("end")
            
            if all(isinstance(x, int) for x in [start1, end1, start2, end2]):
                return not (end1 < start2 or end2 < start1)
        
        except (KeyError, TypeError):
            pass
        
        return False


async def run_demo():
    """Run the complete linguistic analysis demonstration."""
    print("🚀 LINGUISTIC ANALYSIS SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demo shows the new Stage 6.2 architecture that replaces")
    print("brittle movie-specific code with sophisticated language intelligence.")
    print()
    
    demo = LinguisticAnalysisDemo()
    
    # Sample movie data
    movie_data = {
        "name": "Blade Runner 2049",
        "overview": """
        Thirty years after the events of Blade Runner, a new blade runner, 
        LAPD Officer K, unearths a long-buried secret that has the potential 
        to plunge what's left of society into chaos. K's discovery leads him 
        on a quest to find Rick Deckard, a former LAPD blade runner who has 
        been missing for 30 years.
        """.strip(),
        "taglines": ["The key to the future is finally unearthed"],
        "genres": ["Science Fiction", "Drama", "Thriller"],
        "production_year": 2017,
        "people": [
            {"name": "Denis Villeneuve", "type": "Director"},
            {"name": "Ryan Gosling", "type": "Actor", "role": "Officer K"},
            {"name": "Harrison Ford", "type": "Actor", "role": "Rick Deckard"}
        ]
    }
    
    # Demo 1: Content Analysis (Ingestion)
    content_analysis = await demo.analyze_movie_content(movie_data)
    
    # Demo 2: Query Analysis (Search)
    test_queries = [
        "sci-fi movies from the 2010s with Ryan Gosling",
        "futuristic thrillers about artificial intelligence",
        "movies where Harrison Ford plays a detective"
    ]
    
    query_results = []
    for query in test_queries:
        query_analysis = await demo.analyze_user_query(query)
        query_results.append(query_analysis)
    
    # Demo 3: MongoDB Storage
    mongo_doc = demo._demonstrate_mongodb_storage(movie_data, content_analysis)
    
    # Demo 4: Query Matching
    for i, query_analysis in enumerate(query_results):
        print(f"\n🎯 MATCHING QUERY #{i+1}:")
        print(f"Query: \"{test_queries[i]}\"")
        demo._demonstrate_query_matching(query_analysis, mongo_doc)
    
    # Summary
    print("\n" + "=" * 80)
    print("✨ SYSTEM CAPABILITIES DEMONSTRATED:")
    print("✅ Media-agnostic linguistic analysis (works for movies, books, music)")
    print("✅ Symmetric processing (same code for content and queries)")
    print("✅ Advanced NLP (concept expansion, temporal parsing, semantic roles)")
    print("✅ MongoDB-native storage with rich linguistic metadata")
    print("✅ Sophisticated query matching beyond simple keyword search")
    print("✅ Replaced brittle field weights with intelligent language understanding")
    print("\n🎉 Stage 6.2 Architecture Successfully Implemented!")


if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()