#!/usr/bin/env python3
"""
CLI test script for LLMQuestionAnswerPlugin.
Tests the plugin via queue system for real-world validation.
"""

import argparse
import sys
from typing import Dict, Any


import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_test_template import PluginTestCLI, create_base_parser, format_result
from src.worker.task_types import TaskType


class LLMQuestionAnswerTest(PluginTestCLI):
    """Test LLMQuestionAnswerPlugin via queue system."""
    
    def __init__(self):
        super().__init__(
            plugin_name="LLMQuestionAnswerPlugin",
            task_type=TaskType.PLUGIN_EXECUTION
        )


async def main_async():
    """Main CLI entry point."""
    parser = create_base_parser(
        plugin_name="LLMQuestionAnswerPlugin",
        description="Test LLM question answering via queue system"
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (e.g., 'What are the themes of Blade Runner?')"
    )
    
    parser.add_argument(
        "--context",
        help="Context for the question (e.g., movie title, description)"
    )
    
    parser.add_argument(
        "--field-name",
        default="themes",
        help="Field name for context (default: themes)"
    )
    
    parser.add_argument(
        "--media-context",
        default="movie",
        help="Media type context (default: movie)"
    )
    
    args = parser.parse_args()
    

    test_client = LLMQuestionAnswerTest()
    

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
    

    if not args.question:
        print("❌ Error: question required for testing")
        print("Usage: python llm_question_answer_endpoint_test.py 'What are the themes of Blade Runner?'")
        return
    

    test_data = {
        "concept": args.question,
        "field_name": args.field_name,
        "media_context": args.media_context
    }
    
    if args.context:
        test_data["context"] = args.context
    
    print(f"🧪 Testing LLMQuestionAnswerPlugin")
    print(f"   Question: {args.question}")
    print(f"   Field: {args.field_name}")
    print(f"   Media context: {args.media_context}")
    if args.context:
        print(f"   Context: {args.context}")
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