#!/usr/bin/env python3
"""
CLI tool for managing all NLP models in the Jelly project.

Usage:
    python manage_models.py check              # Check model status
    python manage_models.py download           # Download missing models
    python manage_models.py download --force   # Re-download all models
    python manage_models.py status             # Show detailed status
    python manage_models.py cleanup            # Remove unused/orphaned models
    python manage_models.py verify             # Verify model integrity
    python manage_models.py update             # Update all models to latest versions
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.shared.model_manager import ModelManager, ModelStatus
import time


def print_model_status(summary):
    """Print a nice formatted model status."""
    print(f"\n🔍 MODEL STATUS SUMMARY")
    print(f"{'='*50}")
    print(f"Total Models: {summary['total_models']}")
    print(f"Required Models: {summary['required_models']}")
    print(f"Available Models: {summary['available_models']}")
    print(f"Total Storage: {summary['total_size_mb']} MB\n")
    
    # Group by package
    packages = {}
    for model_id, model_info in summary['models'].items():
        package = model_info['package']
        if package not in packages:
            packages[package] = []
        packages[package].append((model_id, model_info))
    
    for package, models in packages.items():
        print(f"📦 {package.upper()}:")
        for model_id, model_info in models:
            status = model_info['status']
            required = "REQUIRED" if model_info['required'] else "OPTIONAL"
            
            if status == "available":
                icon = "✅"
            elif status == "missing":
                icon = "❌"
            elif status == "downloading":
                icon = "⏬"
            else:
                icon = "⚠️ "
            
            print(f"  {icon} {model_info['name']} ({model_info['size_mb']} MB) - {required}")
            
            if model_info['error']:
                print(f"      Error: {model_info['error']}")
        print()


async def main():
    parser = argparse.ArgumentParser(description="Jelly Model Manager")
    parser.add_argument("action", choices=["check", "download", "status", "cleanup", "verify", "update"], 
                       help="Action to perform")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-download all models")
    parser.add_argument("--models-path", default="./models", 
                       help="Path for model storage")
    parser.add_argument("--json", action="store_true", 
                       help="Output in JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive mode with progress bars")
    parser.add_argument("--cleanup-cache", action="store_true", 
                       help="Also cleanup model caches and temp files")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without executing")
    
    args = parser.parse_args()
    
    # Create models directory
    Path(args.models_path).mkdir(parents=True, exist_ok=True)
    
    # Create model manager
    manager = ModelManager(models_base_path=args.models_path)
    
    if args.action == "check":
        print("🔍 Checking model status...")
        await manager.check_all_models()
        summary = manager.get_model_summary()
        
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print_model_status(summary)
    
    elif args.action == "download":
        if args.force:
            print("🔄 Force downloading all models...")
        else:
            print("📥 Downloading missing models...")
        
        success = await manager.download_missing_models(force_download=args.force)
        
        # Show final status
        await manager.check_all_models()
        summary = manager.get_model_summary()
        
        if not args.json:
            print_model_status(summary)
        
        if success:
            print("✅ Model download completed successfully!")
        else:
            print("❌ Some model downloads failed!")
            sys.exit(1)
    
    elif args.action == "cleanup":
        print("🧹 Cleaning up unused models and cache...")
        
        if args.dry_run:
            print("🔍 DRY RUN - No files will be deleted")
        
        cleaned_files, space_freed = await manager.cleanup_models(
            cleanup_cache=args.cleanup_cache, 
            dry_run=args.dry_run
        )
        
        if args.dry_run:
            print(f"Would clean {cleaned_files} files, freeing {space_freed:.1f} MB")
        else:
            print(f"✅ Cleaned {cleaned_files} files, freed {space_freed:.1f} MB")
    
    elif args.action == "verify":
        print("🔍 Verifying model integrity...")
        
        verification_results = await manager.verify_models()
        
        all_valid = True
        for model_id, result in verification_results.items():
            status_icon = "✅" if result['valid'] else "❌"
            print(f"  {status_icon} {model_id}: {result['message']}")
            if not result['valid']:
                all_valid = False
        
        if all_valid:
            print("✅ All models verified successfully!")
        else:
            print("❌ Some models failed verification!")
            sys.exit(1)
    
    elif args.action == "update":
        print("🔄 Updating all models to latest versions...")
        
        if args.dry_run:
            print("🔍 DRY RUN - No models will be updated")
        
        update_results = await manager.update_all_models(dry_run=args.dry_run)
        
        updated_count = sum(1 for result in update_results.values() if result.get('updated', False))
        
        if args.dry_run:
            print(f"Would update {updated_count} models")
        else:
            print(f"✅ Updated {updated_count} models successfully!")
    
    elif args.action == "status":
        print("📊 Getting detailed model status...")
        await manager.check_all_models()
        summary = manager.get_model_summary()
        
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print_model_status(summary)
            
            # Additional status info
            print("🔍 DETAILED INFORMATION:")
            print(f"Model base path: {manager.models_base_path}")
            print(f"NLTK data path: {manager.nltk_data_path}")
            print(f"Gensim data path: {manager.gensim_data_path}")
            print(f"SpaCy data path: {manager.spacy_data_path}")
            
            # Check if paths exist
            paths_to_check = [
                ("Models base", manager.models_base_path),
                ("NLTK data", manager.nltk_data_path),
                ("Gensim data", manager.gensim_data_path),
                ("SpaCy data", manager.spacy_data_path),
            ]
            
            print("\n📁 PATH STATUS:")
            for name, path in paths_to_check:
                exists = "✅" if path.exists() else "❌"
                print(f"  {exists} {name}: {path}")
            
            # Show disk usage if path exists
            if path.exists():
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                print(f"     💾 Storage used: {total_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)