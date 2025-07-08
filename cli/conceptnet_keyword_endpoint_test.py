#!/usr/bin/env python3
"""
CLI test script for ConceptNetKeywordPlugin.
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


class ConceptNetKeywordTest(PluginTestCLI):
    """Test ConceptNetKeywordPlugin via queue system."""
    
    def __init__(self):
        super().__init__(
            plugin_name="ConceptNetKeywordPlugin",
            task_type=TaskType.PLUGIN_EXECUTION
        )


async def main_async():
    """Main CLI entry point (async)."""
    parser = create_base_parser(
        plugin_name="ConceptNetKeywordPlugin",
        description="Test ConceptNet keyword expansion via queue system"
    )
    
    parser.add_argument(
        "keyword",
        nargs="?",
        help="Keyword to expand (e.g., 'science fiction')"
    )
    
    parser.add_argument(
        "--field-name",
        default="genre",
        help="Field name for context (default: genre)"
    )
    
    parser.add_argument(
        "--max-keywords",
        type=int,
        default=10,
        help="Maximum number of keywords to return (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Initialize test client
    test_client = ConceptNetKeywordTest()
    
    # Handle info requests
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
    
    # Require keyword for testing
    if not args.keyword:
        print("❌ Error: keyword required for testing")
        print("Usage: python conceptnet_keyword_endpoint_test.py 'science fiction'")
        return
    
    # Prepare test data - map to router service format
    test_data = {
        "concept": args.keyword,  # Router expects 'concept' field
        "field_name": args.field_name,
        "max_concepts": args.max_keywords  # Router expects 'max_concepts'
    }
    
    print(f"🧪 Testing ConceptNetKeywordPlugin")
    print(f"   Keyword: {args.keyword}")
    print(f"   Field: {args.field_name}")
    print(f"   Max keywords: {args.max_keywords}")
    print()
    
    # Run test
    result = await test_client.test_plugin(test_data, timeout=args.timeout)
    
    # Display result
    print(format_result(result, verbose=args.verbose))
    
    # Exit with appropriate code
    sys.exit(0 if "error" not in result else 1)


def main():
    """Main entry point wrapper."""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()