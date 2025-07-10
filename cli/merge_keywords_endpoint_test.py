#!/usr/bin/env python3
"""
CLI test script for MergeKeywordsPlugin.
Tests the plugin via queue system for real-world validation.
"""

import argparse
import sys
from typing import Dict, Any


import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_test_template import PluginTestCLI, create_base_parser, format_result
from src.worker.task_types import TaskType


class MergeKeywordsTest(PluginTestCLI):
    """Test MergeKeywordsPlugin via queue system."""
    
    def __init__(self):
        super().__init__(
            plugin_name="MergeKeywordsPlugin",
            task_type=TaskType.PLUGIN_EXECUTION
        )


async def main_async():
    """Main CLI entry point."""
    parser = create_base_parser(
        plugin_name="MergeKeywordsPlugin",
        description="Test keyword merging via queue system"
    )
    
    parser.add_argument(
        "keyword",
        nargs="?",
        help="Keyword to expand and merge (e.g., 'science fiction')"
    )
    
    parser.add_argument(
        "--field-name",
        default="genre",
        help="Field name for context (default: genre)"
    )
    
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["conceptnet", "llm", "gensim"],
        help="List of providers to use (default: conceptnet llm gensim)"
    )
    
    parser.add_argument(
        "--merge-strategy",
        default="union",
        choices=["union", "intersection", "weighted", "ranked"],
        help="Merge strategy (default: union)"
    )
    
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=25,
        help="Maximum number of merged keywords (default: 25)"
    )
    
    args = parser.parse_args()
    

    test_client = MergeKeywordsTest()
    

    if args.stats:
        stats = await test_client.get_queue_stats()
        print(f"Queue Statistics:")
        print(f"  CPU pending: {stats['queues']['cpu_pending']}")
        print(f"  GPU pending: {stats['queues']['gpu_pending']}")
        print(f"  Failed tasks: {stats['queues']['failed_tasks']}")
        print(f"  Connected clients: {stats['redis_info'].get('connected_clients', 'N/A')}")
        return
    
    if args.definition:
        definition = test_client.get_task_definition()
        if definition:
            print(f"Task Definition:")
            print(f"  Plugin: {definition['plugin_name']}")
            print(f"  Description: {definition['description']}")
            print(f"  Required fields: {definition['required_fields']}")
            print(f"  Optional fields: {definition['optional_fields']}")
            print(f"  Timeout: {definition['execution_timeout']}s")
            print(f"  Requires service: {definition['requires_service']}")
            if definition['service_type']:
                print(f"  Service type: {definition['service_type']}")
        return
    

    if not args.keyword:
        print("❌ Error: keyword required for testing")
        print("Usage: python merge_keywords_endpoint_test.py 'science fiction'")
        return
    

    test_data = {
        "concept": args.keyword,
        "field_name": args.field_name,
        "providers": args.providers,
        "merge_strategy": args.merge_strategy,
        "max_concepts": args.max_keywords
    }
    
    print(f"🧪 Testing MergeKeywordsPlugin")
    print(f"   Keyword: {args.keyword}")
    print(f"   Field: {args.field_name}")
    print(f"   Providers: {', '.join(args.providers)}")
    print(f"   Merge strategy: {args.merge_strategy}")
    print(f"   Max keywords: {args.max_keywords}")
    print()
    

    result = await test_client.test_plugin(test_data, timeout=args.timeout)
    

    print(format_result(result, verbose=args.verbose))
    

    sys.exit(0 if "error" not in result else 1)


def main():
    """Main entry point wrapper."""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()