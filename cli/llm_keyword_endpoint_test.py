#!/usr/bin/env python3
"""
CLI test script for LLMKeywordPlugin.
Tests the plugin via queue system for real-world validation.
"""

import argparse
import sys
from typing import Dict, Any


import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_test_template import PluginTestCLI, create_base_parser, format_result
from src.worker.task_types import TaskType


class LLMKeywordTest(PluginTestCLI):
    """Test LLMKeywordPlugin via queue system."""
    
    def __init__(self):
        super().__init__(
            plugin_name="LLMKeywordPlugin",
            task_type=TaskType.PLUGIN_EXECUTION
        )


async def main_async():
    """Main CLI entry point."""
    parser = create_base_parser(
        plugin_name="LLMKeywordPlugin",
        description="Test LLM keyword expansion via queue system"
    )
    
    parser.add_argument(
        "keyword",
        nargs="?",
        help="Keyword to expand (e.g., 'psychological thriller')"
    )
    
    parser.add_argument(
        "--field-name",
        default="genre",
        help="Field name for context (default: genre)"
    )
    
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=15,
        help="Maximum number of keywords to return (default: 15)"
    )
    
    parser.add_argument(
        "--media-context",
        default="movie",
        help="Media type context (default: movie)"
    )
    
    args = parser.parse_args()
    

    test_client = LLMKeywordTest()
    

    if args.stats:
        stats = test_client.get_queue_stats()
        print(f"Queue Statistics:")
        print(f"  Pending tasks: {stats['pending_tasks']}")
        print(f"  Failed tasks: {stats['failed_tasks']}")
        print(f"  Connected clients: {stats['redis_info']['connected_clients']}")
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
        print("Usage: python llm_keyword_endpoint_test.py 'psychological thriller'")
        return
    

    test_data = {
        "concept": args.keyword,
        "field_name": args.field_name,
        "max_concepts": args.max_keywords,
        "media_context": args.media_context
    }
    
    print(f"🧪 Testing LLMKeywordPlugin")
    print(f"   Keyword: {args.keyword}")
    print(f"   Field: {args.field_name}")
    print(f"   Max keywords: {args.max_keywords}")
    print(f"   Media context: {args.media_context}")
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