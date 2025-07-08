#!/usr/bin/env python3
"""
CLI test script for LLMTemporalIntelligencePlugin.
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


class LLMTemporalIntelligenceTest(PluginTestCLI):
    """Test LLMTemporalIntelligencePlugin via queue system."""
    
    def __init__(self):
        super().__init__(
            plugin_name="LLMTemporalIntelligencePlugin",
            task_type=TaskType.PLUGIN_EXECUTION
        )


async def main_async():
    """Main CLI entry point."""
    parser = create_base_parser(
        plugin_name="LLMTemporalIntelligencePlugin",
        description="Test LLM temporal intelligence via queue system"
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze for temporal concepts (e.g., 'A story about the Cold War era and its aftermath')"
    )
    
    parser.add_argument(
        "--field-name",
        default="plot",
        help="Field name for context (default: plot)"
    )
    
    parser.add_argument(
        "--media-context",
        default="movie",
        help="Media type context (default: movie)"
    )
    
    parser.add_argument(
        "--analysis-type",
        default="comprehensive",
        choices=["comprehensive", "periods", "events", "themes"],
        help="Type of temporal analysis (default: comprehensive)"
    )
    
    args = parser.parse_args()
    
    # Initialize test client
    test_client = LLMTemporalIntelligenceTest()
    
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
    
    # Require text for testing
    if not args.text:
        print("❌ Error: text required for testing")
        print("Usage: python llm_temporal_intelligence_endpoint_test.py 'A story about the Cold War era and its aftermath'")
        return
    
    # Prepare test data - map to router service format
    test_data = {
        "concept": args.text,  # Router expects 'concept' field
        "field_name": args.field_name,
        "media_context": args.media_context,
        "analysis_type": args.analysis_type
    }
    
    print(f"🧪 Testing LLMTemporalIntelligencePlugin")
    print(f"   Text: {args.text}")
    print(f"   Field: {args.field_name}")
    print(f"   Media context: {args.media_context}")
    print(f"   Analysis type: {args.analysis_type}")
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