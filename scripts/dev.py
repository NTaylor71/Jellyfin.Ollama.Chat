import argparse
import asyncio
import sys

def run(func):
    try:
        asyncio.run(func())
    except KeyboardInterrupt:
        print("\n👋 Cancelled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS RAG Dev CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("cli", help="Run interactive chat client")
    sub.add_parser("test-chat", help="Test a roundtrip query only")
    sub.add_parser("test-ingest-query", help="Full ingest and query test")

    args = parser.parse_args()

    if args.cmd == "cli":
        from scripts.cli_send import main as cli_main
        run(cli_main)

    elif args.cmd == "test-chat":
        from tests.test_chat import test_roundtrip
        run(test_roundtrip)

    elif args.cmd == "test-ingest-query":
        from tests.test_end_to_end import ingest_sample, query_sample
        run(ingest_sample)
        run(query_sample)

    else:
        parser.print_help()
        sys.exit(1)
