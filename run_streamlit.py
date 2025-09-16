#!/usr/bin/env python3
"""
Streamlit Launcher for Diabetic Retinopathy AI System
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    
    print("🚀 Launching Diabetic Retinopathy AI System - Streamlit App")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} is installed")
    except ImportError:
        print("❌ Streamlit is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
    
    # Check if other dependencies are installed
    required_packages = ["pandas", "numpy", "plotly", "matplotlib", "seaborn"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ All packages installed successfully")
    
    # Launch Streamlit app
    print("\n🌐 Launching Streamlit application...")
    print("The app will open in your default web browser")
    print("Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit application stopped by user")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("Try running manually: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 