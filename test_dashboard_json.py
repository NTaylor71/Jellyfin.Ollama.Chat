#!/usr/bin/env python3
"""
Test JSON validity of Grafana dashboard
"""

import json
import sys

def test_json_file(filepath):
    """Test if JSON file is valid."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✓ JSON file {filepath} is valid")
        print(f"  Dashboard title: {data.get('dashboard', {}).get('title', 'Unknown')}")
        print(f"  Panel count: {len(data.get('dashboard', {}).get('panels', []))}")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ JSON file {filepath} is invalid:")
        print(f"  Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading {filepath}: {e}")
        return False

if __name__ == "__main__":
    dashboard_file = "docker/monitoring/grafana/provisioning/dashboards/json/plugin-performance.json"
    existing_dashboard = "docker/monitoring/grafana/provisioning/dashboards/json/production-rag-api.json"
    
    print("Testing dashboard JSON files...")
    print()
    
    # Test our new dashboard
    print("1. Testing new plugin performance dashboard:")
    result1 = test_json_file(dashboard_file)
    print()
    
    # Test existing dashboard for comparison
    print("2. Testing existing production dashboard:")
    result2 = test_json_file(existing_dashboard)
    print()
    
    if result1 and result2:
        print("✓ Both dashboard files are valid JSON")
    elif result1:
        print("⚠ New dashboard is valid, but existing dashboard has issues")
    elif result2:
        print("⚠ Existing dashboard is valid, but new dashboard has issues")
    else:
        print("✗ Both dashboard files have issues")