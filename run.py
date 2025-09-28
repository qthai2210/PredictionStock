"""
Simple runner script for Vietnamese Stock Analysis
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if requirements are installed"""
    try:
        import vnstock
        import pandas
        import numpy
        return True
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        return False

def run_setup():
    """Run setup if needed"""
    print("🔧 Running first-time setup...")
    try:
        result = subprocess.run([sys.executable, 'setup.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Setup completed successfully")
            return True
        else:
            print(f"❌ Setup failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running setup: {e}")
        return False

def main():
    """Run the main application"""
    print("🇻🇳 Starting Vietnamese Stock Analysis Tool...")
    print("=" * 50)
    
    # Check if this is first run
    if not os.path.exists('cache') or not check_requirements():
        print("\n⚠️  First time setup or missing dependencies detected.")
        
        setup_choice = input("Run setup now? (y/n): ").strip().lower()
        if setup_choice == 'y':
            if not run_setup():
                print("Setup failed. Please install requirements manually:")
                print("  pip install -r requirements.txt")
                return
        else:
            print("Continuing without setup...")
    
    # Run main application
    print("\n🚀 Launching application...")
    try:
        subprocess.run([sys.executable, 'main.py'])
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main()
