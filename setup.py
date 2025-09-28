"""
Setup script for Vietnamese Stock Analysis Project
Run this script to install dependencies and set up the project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Successfully installed all requirements!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'cache',
        'data',
        'models',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def test_installation():
    """Test if vnstock is properly installed"""
    print("\n🧪 Testing vnstock installation...")
    
    try:
        from vnstock import Quote
        print("✅ vnstock imported successfully!")
        
        # Test basic functionality
        quote = Quote(symbol='VNM', source='VCI')
        print("✅ Quote object created successfully!")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import vnstock: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        return True

def main():
    """Main setup function"""
    print("🇻🇳 Vietnamese Stock Analysis Project Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed due to package installation issues")
        return
    
    # Test installation
    if test_installation():
        print("\n✅ Setup completed successfully!")
        print("\n🚀 To run the application:")
        print("   python main.py")
        print("\n📚 To run tests:")
        print("   python -m pytest tests/")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()