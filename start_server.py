#!/usr/bin/env python3
"""
Simple Phone Fingerprint Server Starter
This script starts the web API server for phone fingerprint integration.
"""

import sys
import os
import subprocess

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "flask", "flask-cors", "numpy", "opencv-python", 
                              "Pillow", "scipy", "scikit-learn"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def find_project_directory():
    # Start from current directory and search up
    current_dir = os.getcwd()
    
    # Check current directory
    if os.path.exists(os.path.join(current_dir, 'phone_fingerprint')):
        return current_dir
    
    # Check parent directories
    parent_dir = os.path.dirname(current_dir)
    while parent_dir != current_dir:
        if os.path.exists(os.path.join(parent_dir, 'phone_fingerprint')):
            return parent_dir
        current_dir = parent_dir
        parent_dir = os.path.dirname(current_dir)
    
    # If not found, try common locations
    common_paths = [
        "D:\\[000] DOST\\regunit",
        "D:\\regunit", 
        "C:\\regunit",
        os.path.expanduser("~/regunit")
    ]
    
    for path in common_paths:
        if os.path.exists(os.path.join(path, 'phone_fingerprint')):
            return path
    
    return None

def start_server():
    """Start the web API server"""
    print("🚀 Starting Phone Fingerprint Web API Server...")
    print("="*60)
    
    project_dir = find_project_directory()
    if not project_dir:
        print("❌ Could not find the phone_fingerprint project directory")
        print("Please make sure you're in the correct directory")
        return False
    
    print(f"📁 Found project at: {project_dir}")
    
    phone_fingerprint_path = os.path.join(project_dir, 'phone_fingerprint')
    sys.path.insert(0, phone_fingerprint_path)
    
    try:
        # Import and start the web server
        from web_api_server import start_server as start_web_server
        start_web_server(host='0.0.0.0', port=5000, debug=False)
        return True
    except ImportError as e:
        print(f"❌ Error importing web server: {e}")
        print("Make sure the phone_fingerprint directory contains web_api_server.py")
        return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    """Main function"""
    print("="*60)
    print("📱 PHONE FINGERPRINT WEB API SERVER")
    print("="*60)
    print()
    print("This will start a web server that can receive")
    print("fingerprint data from your phone's capacitive sensor.")
    print()
    

    try:
        import flask
        import flask_cors
        print("✅ Dependencies already installed")
    except ImportError:
        print("📦 Installing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install manually:")
            print("   pip install flask flask-cors numpy opencv-python Pillow scipy scikit-learn")
            return
    
    print()
    print("🔧 Starting server...")
    print()
    
    if start_server():
        print("✅ Server started successfully!")
    else:
        print("❌ Failed to start server")

if __name__ == "__main__":
    main() 