#!/usr/bin/env python3
"""
Phone Fingerprint System Startup Script
This script helps you start the phone fingerprint system with web API integration.
"""

import sys
import os
import subprocess
import webbrowser
import time

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

def get_local_ip():
    """Get local IP address"""
    import socket
    try:
        # Get local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def start_web_server():
    """Start the web API server"""
    print("🚀 Starting Phone Fingerprint Web API Server...")
    print("="*60)
    
    local_ip = get_local_ip()
    server_url = f"http://{local_ip}:5000"
    
    print(f"🌐 Server will be available at: {server_url}")
    print(f"📱 Mobile app URL: {server_url}/mobile_app_example.html")
    print()
    print("📋 Instructions:")
    print("1. The server will start on your computer")
    print("2. Open the mobile app URL on your phone")
    print("3. Use your phone's fingerprint sensor")
    print("4. See all processing on your computer screen")
    print()
    
    # Add phone_fingerprint to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phone_fingerprint'))
    
    try:
        # Import and start the web server
        from web_api_server import start_server
        start_server(host='0.0.0.0', port=5000, debug=False)
    except ImportError as e:
        print(f"❌ Error importing web server: {e}")
        print("Make sure you're in the correct directory")
        return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    """Main function"""
    print("="*60)
    print("📱 PHONE FINGERPRINT SYSTEM SETUP")
    print("="*60)
    print()
    print("This will set up a web API server that can receive")
    print("fingerprint data from your phone's capacitive sensor.")
    print()
    
    # Check if dependencies are installed
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
    print("🔧 Setup Options:")
    print("1. Start Web API Server (Recommended)")
    print("2. Run Test Mode (Simulation)")
    print("3. Run Main System (Interactive Menu)")
    print("4. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting Web API Server...")
            start_web_server()
            break
        elif choice == '2':
            print("\n🧪 Starting Test Mode...")
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phone_fingerprint'))
                from phone_integration_example import main as test_main
                test_main()
            except Exception as e:
                print(f"❌ Error: {e}")
            break
        elif choice == '3':
            print("\n📱 Starting Main System...")
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phone_fingerprint'))
                from phone_main import main as main_system
                main_system()
            except Exception as e:
                print(f"❌ Error: {e}")
            break
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main() 