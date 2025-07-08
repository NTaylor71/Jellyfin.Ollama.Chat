#!/usr/bin/env python3
"""
Simple script to clear the concept expansion cache.
Usage: python clear_cache.py [--all|--conceptnet|--pattern PATTERN]
"""

import asyncio
import argparse
from src.api.cache_admin import get_cache_admin, print_cache_summary

async def main():
    parser = argparse.ArgumentParser(description="Clear concept expansion cache")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Clear all cache entries")
    group.add_argument("--conceptnet", action="store_true", help="Clear only ConceptNet entries")
    group.add_argument("--pattern", type=str, help="Clear entries matching regex pattern")
    parser.add_argument("--summary", action="store_true", help="Show cache summary after operation")
    
    args = parser.parse_args()
    
    admin = get_cache_admin()
    
    if args.all:
        print("🧹 Clearing ALL cache entries...")
        result = await admin.clear_all_cache()
        print(f"✅ {result['message']}")
    elif args.conceptnet:
        print("🧹 Clearing ConceptNet cache entries...")
        result = await admin.clear_cache_by_method("conceptnet")
        print(f"✅ {result['message']}")
    elif args.pattern:
        print(f"🧹 Clearing cache entries matching pattern '{args.pattern}'...")
        result = await admin.clear_cache_by_pattern(args.pattern)
        print(f"✅ {result['message']}")
    else:
        print("📊 Cache Summary:")
        await print_cache_summary()
        print("\nUsage: python clear_cache.py --all|--conceptnet|--pattern PATTERN")
        return
    
    if args.summary:
        print("\n📊 Cache Summary after operation:")
        await print_cache_summary()

if __name__ == "__main__":
    asyncio.run(main())