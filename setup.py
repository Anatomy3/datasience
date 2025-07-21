#!/usr/bin/env python3
"""
Quick Setup Script for DS Salaries Dashboard
Creates necessary folders and files
"""

import os
import sys

def create_folders():
    """Create necessary folders"""
    folders = [
        'assets',
        'data', 
        'components',
        'pages',
        'utils',
        'exports',
        'exports/reports'
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Created folder: {folder}")
        else:
            print(f"📁 Folder exists: {folder}")

def create_init_files():
    """Create __init__.py files"""
    init_files = [
        'components/__init__.py',
        'pages/__init__.py', 
        'utils/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Module initialization\n')
            print(f"✅ Created: {init_file}")
        else:
            print(f"📄 File exists: {init_file}")

def check_data_files():
    """Check if data files exist"""
    data_files = [
        'data/ds_salaries.csv',
        'data/model.pkl'
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✅ Data file found: {data_file}")
        else:
            print(f"⚠️  Missing: {data_file} - Please copy your file here")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'dash',
        'dash_bootstrap_components', 
        'plotly',
        'pandas',
        'scikit-learn',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ Package installed: {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ Missing package: {package}")
    
    if missing_packages:
        print(f"\n🔧 To install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
    
    return len(missing_packages) == 0

def main():
    """Main setup function"""
    print("🚀 DS Salaries Dashboard - Quick Setup")
    print("=" * 50)
    
    print("\n📁 Creating folders...")
    create_folders()
    
    print("\n📄 Creating __init__.py files...")
    create_init_files()
    
    print("\n📊 Checking data files...")
    check_data_files()
    
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 50)
    
    if deps_ok:
        print("✅ Setup complete! You can now run:")
        print("   python app.py")
    else:
        print("⚠️  Please install missing dependencies first:")
        print("   pip install -r requirements.txt")
    
    print("\n📝 Don't forget to:")
    print("   1. Copy ds_salaries.csv to data/ folder")
    print("   2. Copy model.pkl to data/ folder (optional)")
    print("   3. Run: python app.py")
    print("   4. Open: http://127.0.0.1:8050")

if __name__ == "__main__":
    main()