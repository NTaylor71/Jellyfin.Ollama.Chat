#!/usr/bin/env python3
"""
CLI test script for LLMSemanticChunkingPlugin.
Tests the plugin via queue system for real-world validation.
"""

import argparse
import sys
from typing import Dict, Any


import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugin_test_template import PluginTestCLI, create_base_parser, format_result
from src.worker.task_types import TaskType


class LLMSemanticChunkingTest(PluginTestCLI):
    """Test LLMSemanticChunkingPlugin via queue system."""
    
    def __init__(self):
        super().__init__(
            plugin_name="LLMSemanticChunkingPlugin",
            task_type=TaskType.PLUGIN_EXECUTION
        )


async def main_async():
    """Main CLI entry point."""
    parser = create_base_parser(
        plugin_name="LLMSemanticChunkingPlugin",
        description="Test LLM semantic chunking via queue system"
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to chunk (or use --file to load from file)"
    )
    
    parser.add_argument(
        "--file",
        help="File containing text to chunk (e.g. todo.5.9.3.md)"
    )
    
    parser.add_argument(
        "--strategy",
        default="semantic_paragraphs",
        choices=["semantic_paragraphs", "sentence_based", "topic_based", "narrative_flow"],
        help="Chunking strategy (default: semantic_paragraphs)"
    )
    
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=500,
        help="Maximum chunk size in words (default: 500)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks in words (default: 50)"
    )
    
    parser.add_argument(
        "--field-name",
        default="content",
        help="Field name for context (default: content)"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Minimum text length to trigger chunking (default: 1000)"
    )
    
    parser.add_argument(
        "--optimize-chunks",
        action="store_true",
        help="Enable chunk optimization using LLM feedback"
    )
    
    parser.add_argument(
        "--superman-review",
        action="store_true",
        help="Test with Superman review from todo.5.9.3.md"
    )
    
    args = parser.parse_args()
    
    # Initialize test client
    test_client = LLMSemanticChunkingTest()
    
    # Handle stats request
    if args.stats:
        stats = await test_client.get_queue_stats()
        print(f"Queue Statistics:")
        print(f"  CPU pending: {stats['queues']['cpu_pending']}")
        print(f"  GPU pending: {stats['queues']['gpu_pending']}")
        print(f"  Failed tasks: {stats['queues']['failed_tasks']}")
        print(f"  Connected clients: {stats['redis_info'].get('connected_clients', 'N/A')}")
        return
    
    # Handle definition request
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
    
    # Get text to chunk
    text_to_chunk = None
    
    if args.superman_review:
        # Load Superman review from todo.5.9.3.md
        try:
            with open("todo.5.9.3.md", "r") as f:
                content = f.read()
                # Extract the Superman review (starts around line 173)
                lines = content.split('\n')
                review_start = None
                for i, line in enumerate(lines):
                    if "superman_review" in line:
                        review_start = i + 1
                        break
                
                if review_start:
                    # Take the review content
                    review_lines = lines[review_start:]
                    # Find the end (empty line or next section)
                    review_end = len(review_lines)
                    for i, line in enumerate(review_lines):
                        if line.strip() == "" and i > 10:  # Skip early empty lines
                            review_end = i
                            break
                    
                    text_to_chunk = '\n'.join(review_lines[:review_end])
                    print(f"📖 Loaded Superman review: {len(text_to_chunk)} characters")
                else:
                    print("❌ Could not find Superman review in todo.5.9.3.md")
                    return
        except FileNotFoundError:
            print("❌ todo.5.9.3.md not found")
            return
    elif args.file:
        # Load text from file
        try:
            with open(args.file, "r") as f:
                text_to_chunk = f.read()
                print(f"📖 Loaded text from {args.file}: {len(text_to_chunk)} characters")
        except FileNotFoundError:
            print(f"❌ File {args.file} not found")
            return
    elif args.text:
        text_to_chunk = args.text
    else:
        print("❌ Error: text required for testing")
        print("Usage: python llm_semantic_chunking_endpoint_test.py 'Long text to chunk...'")
        print("   or: python llm_semantic_chunking_endpoint_test.py --file todo.5.9.3.md")
        print("   or: python llm_semantic_chunking_endpoint_test.py --superman-review")
        return
    
    # Prepare test data
    test_data = {
        "concept": text_to_chunk,
        "field_name": args.field_name,
        "media_context": "text",
        "config": {
            "strategy": args.strategy,
            "max_chunk_size": args.max_chunk_size,
            "overlap": args.overlap,
            "min_length": args.min_length,
            "optimize_chunks": args.optimize_chunks
        }
    }
    
    print(f"🧪 Testing LLMSemanticChunkingPlugin")
    print(f"   Strategy: {args.strategy}")
    print(f"   Max chunk size: {args.max_chunk_size} words")
    print(f"   Overlap: {args.overlap} words")
    print(f"   Text length: {len(text_to_chunk)} characters")
    print(f"   Field: {args.field_name}")
    print(f"   Optimize: {args.optimize_chunks}")
    print()
    
    # Test the plugin
    result = await test_client.test_plugin(test_data, timeout=args.timeout)
    
    # Format and display results
    if args.verbose:
        print(format_result(result, verbose=True))
    else:
        # Custom formatting for chunking results
        print("📊 Chunking Results:")
        print("=" * 50)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            plugin_result = result.get("plugin_result", {})
            chunks = plugin_result.get("chunks", [])
            metadata = plugin_result.get("metadata", {})
            
            if chunks:
                print(f"✅ Successfully created {len(chunks)} chunks")
                print(f"   Strategy: {metadata.get('strategy', 'unknown')}")
                print(f"   Original length: {metadata.get('original_length', 0)} chars")
                print(f"   Average chunk size: {metadata.get('avg_chunk_size', 0):.1f} words")
                print()
                
                # Show chunk previews
                for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                    chunk_type = chunk.get("type", "unknown")
                    word_count = chunk.get("word_count", 0)
                    preview = chunk.get("text", "")[:100] + "..." if len(chunk.get("text", "")) > 100 else chunk.get("text", "")
                    
                    print(f"Chunk {i+1} ({chunk_type}, {word_count} words):")
                    print(f"  {preview}")
                    print()
                
                if len(chunks) > 5:
                    print(f"... and {len(chunks) - 5} more chunks")
                    print()
                
                # Show metadata
                print("📋 Metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            else:
                print("❌ No chunks created")
    
    # Exit with appropriate code
    sys.exit(0 if "error" not in result else 1)


def main():
    """Main entry point wrapper."""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()