#!/usr/bin/env python3
"""
CLI test script for LLMWebSearchPlugin.
Tests the plugin via queue system for real-world validation.
"""

import argparse
import sys
from typing import Dict, Any

# Add project root to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_test_template import PluginTestCLI, create_base_parser, format_result
from src.worker.task_types import TaskType


class LLMWebSearchTest(PluginTestCLI):
    """Test LLMWebSearchPlugin via queue system."""
    
    def __init__(self):
        super().__init__(
            plugin_name="LLMWebSearchPlugin",
            task_type=TaskType.PLUGIN_EXECUTION
        )


async def main_async():
    """Main CLI entry point."""
    parser = create_base_parser(
        plugin_name="LLMWebSearchPlugin",
        description="Test LLM web search + processing via queue system"
    )
    
    parser.add_argument(
        "movie_name",
        nargs="?",
        help="Movie name to search for (e.g., 'The Matrix')"
    )
    
    parser.add_argument(
        "--year",
        type=int,
        help="Movie production year (e.g., 1999)"
    )
    
    parser.add_argument(
        "--search-type",
        choices=["reviews", "box_office", "production", "awards", "cultural"],
        default="reviews",
        help="Type of web search to perform (default: reviews)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum results per search template (default: 5)"
    )
    
    parser.add_argument(
        "--domains",
        nargs="*",
        help="Specific domains to search (e.g., variety.com imdb.com)"
    )
    
    args = parser.parse_args()

    # Set up movie data based on user input
    movie_name = args.movie_name or "The Matrix"
    production_year = args.year or 1999
    
    print(f"🎬 Testing LLM Web Search for: {movie_name} ({production_year})")
    print(f"📊 Search type: {args.search_type}")
    print(f"🔍 Max results per template: {args.max_results}")
    if args.domains:
        print(f"🌐 Filtering domains: {', '.join(args.domains)}")

    # Define search configurations for different types
    search_configs = {
        "reviews": {
            "search_templates": [
                {
                    "template": "professional film review {Name} {ProductionYear} critic",
                    "max_results": args.max_results,
                    "domains": args.domains or ["variety.com", "theguardian.com", "nytimes.com"]
                },
                {
                    "template": "audience reaction {Name} movie reddit imdb user review",
                    "max_results": args.max_results,
                    "domains": args.domains or ["reddit.com", "imdb.com", "letterboxd.com"]
                }
            ],
            "llm_processing": {
                "prompt": """Analyze these real web reviews for "{Name}" ({ProductionYear}):
{search_results}

Extract and synthesize:
1. Critical reception consensus and patterns
2. Audience sentiment trends and reactions  
3. Specific praise points mentioned repeatedly
4. Common criticism themes
5. Overall reception score/rating patterns""",
                "expected_format": "dict",
                "fields": ["critical_consensus", "audience_sentiment", "praise_themes", "criticism_themes", "rating_analysis"]
            }
        },
        
        "box_office": {
            "search_templates": [
                {
                    "template": "box office {Name} {ProductionYear} opening weekend gross earnings",
                    "max_results": args.max_results,
                    "domains": args.domains or ["boxofficemojo.com", "variety.com", "deadline.com"]
                },
                {
                    "template": "{Name} movie budget production cost marketing",
                    "max_results": args.max_results,
                    "domains": args.domains or ["variety.com", "hollywoodreporter.com"]
                }
            ],
            "llm_processing": {
                "prompt": """Extract commercial performance data for "{Name}" ({ProductionYear}):
{search_results}

Focus on factual financial information:
1. Box office numbers (opening weekend, total gross, international)
2. Production budget and costs
3. Marketing campaign scale and effectiveness
4. Profitability analysis and studio performance""",
                "expected_format": "dict",
                "fields": ["box_office_performance", "budget_analysis", "marketing_impact", "profitability"]
            }
        },
        
        "production": {
            "search_templates": [
                {
                    "template": "making of {Name} behind scenes trivia filming stories",
                    "max_results": args.max_results
                },
                {
                    "template": "{Name} cast interviews director production process",
                    "max_results": args.max_results
                }
            ],
            "llm_processing": {
                "prompt": """Compile behind-the-scenes information for "{Name}":
{search_results}

Organize factual production information:
1. Notable production challenges and solutions
2. Casting stories and actor preparation
3. Director's vision and creative process insights
4. Technical innovations or filming techniques used""",
                "expected_format": "dict",
                "fields": ["production_challenges", "casting_stories", "director_insights", "technical_innovations"]
            }
        },
        
        "awards": {
            "search_templates": [
                {
                    "template": "{Name} {ProductionYear} awards nominations Oscar Emmy Golden Globe",
                    "max_results": args.max_results,
                    "domains": args.domains or ["oscars.org", "goldenglobes.com", "variety.com"]
                },
                {
                    "template": "{Name} film festival awards Cannes Venice Sundance",
                    "max_results": args.max_results,
                    "domains": args.domains or ["festival-cannes.com", "variety.com", "indiewire.com"]
                }
            ],
            "llm_processing": {
                "prompt": """Compile awards and recognition data for "{Name}":
{search_results}

Document factual recognition received:
1. Major awards won (Oscars, Golden Globes, BAFTAs, etc.)
2. Nominations received across all categories
3. Film festival selections and awards
4. Industry guild recognitions""",
                "expected_format": "dict",
                "fields": ["major_awards", "nominations_received", "festival_recognition", "guild_honors"]
            }
        },
        
        "cultural": {
            "search_templates": [
                {
                    "template": "{Name} cultural impact influence cinema later films legacy",
                    "max_results": args.max_results
                },
                {
                    "template": "{Name} memes quotes popular culture references internet",
                    "max_results": args.max_results,
                    "domains": args.domains or ["knowyourmeme.com", "reddit.com"]
                }
            ],
            "llm_processing": {
                "prompt": """Assess the real cultural legacy of "{Name}" ({ProductionYear}):
{search_results}

Evaluate documented cultural impact:
1. Lasting influence on cinema and filmmaking
2. Popular culture penetration (memes, quotes, parodies)
3. Critical reevaluation and retrospective analysis
4. Academic or scholarly discussion and study""",
                "expected_format": "dict",
                "fields": ["cinema_influence", "pop_culture_penetration", "critical_reevaluation", "academic_discussion"]
            }
        }
    }

    # Get configuration for selected search type
    config = search_configs[args.search_type]
    
    # Add template variables for substitution
    config["template_variables"] = {
        "Name": movie_name,
        "ProductionYear": production_year,
        "Title": movie_name
    }
    
    # Add rate limiting
    config["rate_limiting"] = {
        "delay_between_searches": 2.0
    }

    # Create test instance
    test = LLMWebSearchTest()
    
    # Prepare field value (dummy for web search)
    field_value = f"{movie_name} ({production_year})"
    
    print(f"\n🚀 Executing web search + LLM processing...")
    print(f"📋 Configuration: {len(config['search_templates'])} search templates")
    
    # Execute the test
    result = await test.test_plugin(
        {
            "field_name": "web_search_test",
            "field_value": field_value,
            "config": config
        },
        timeout=args.timeout if hasattr(args, 'timeout') else 60
    )

    # Format and display results
    print("\n" + "="*80)
    print("🎯 WEB SEARCH + LLM PROCESSING RESULTS")
    print("="*80)
    
    if result.get("success", False):
        websearch_data = result.get("result", {})
        
        # Display search results summary
        search_results = websearch_data.get("websearch_results", [])
        print(f"\n📊 Search Results Summary:")
        print(f"   Total results found: {len(search_results)}")
        
        if search_results:
            print(f"   Sources found:")
            sources = set()
            for res in search_results:
                if res.get("url"):
                    domain = res["url"].split("//")[-1].split("/")[0]
                    sources.add(domain)
            for source in sorted(sources):
                print(f"     - {source}")
        
        # Display LLM analysis
        llm_analysis = websearch_data.get("llm_analysis", {})
        if llm_analysis:
            print(f"\n🤖 LLM Analysis:")
            for field, content in llm_analysis.items():
                if content:
                    print(f"\n  {field.replace('_', ' ').title()}:")
                    if isinstance(content, str):
                        print(f"    {content[:300]}{'...' if len(content) > 300 else ''}")
                    elif isinstance(content, list):
                        for item in content[:3]:  # Show first 3 items
                            print(f"    • {item}")
                        if len(content) > 3:
                            print(f"    ... and {len(content) - 3} more")
                    else:
                        print(f"    {str(content)[:200]}{'...' if len(str(content)) > 200 else ''}")
        
        # Display metadata
        metadata = websearch_data.get("search_metadata", {})
        if metadata:
            print(f"\n📈 Execution Metadata:")
            print(f"   Execution time: {metadata.get('execution_time_ms', 0):.1f}ms")
            print(f"   Templates used: {len(metadata.get('templates_used', []))}")
            print(f"   Queries executed: {len(metadata.get('queries_executed', []))}")
            
            if hasattr(args, 'debug') and args.debug and metadata.get('queries_executed'):
                print(f"\n🔍 Queries executed:")
                for i, query in enumerate(metadata['queries_executed'], 1):
                    print(f"     {i}. {query}")
    else:
        print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
        if hasattr(args, 'debug') and args.debug:
            print(f"\nDebug info: {format_result(result)}")

    print("\n" + "="*80)


def main():
    """Synchronous main entry point."""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()