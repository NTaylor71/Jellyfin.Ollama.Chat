#!/usr/bin/env python3
"""
Launch script for the Queue and Resource Monitor App
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import PyQt6
        import GPUtil
        import psutil
        import redis
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def install_dependencies():
    """Install required dependencies using uv"""
    print("Installing GUI dependencies...")
    try:
        subprocess.run([
            "uv", "add", "PyQt6>=6.4.0", "GPUtil>=1.4.0"
        ], check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def main():
    """Main launcher function"""
    print("Queue & Resource Monitor App Launcher")
    print("=" * 40)
    

    if not check_dependencies():
        print("\nRequired dependencies not found.")
        response = input("Would you like to install them? (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                print("Failed to install dependencies. Exiting.")
                sys.exit(1)
        else:
            print("Cannot run without dependencies. Exiting.")
            sys.exit(1)
    

    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    

    parent_dir = os.path.dirname(app_dir)
    sys.path.insert(0, parent_dir)
    
    print("\nStarting Queue & Resource Monitor App...")
    print("Press Ctrl+C to stop")
    

    try:
        from main import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()