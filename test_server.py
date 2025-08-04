#!/usr/bin/env python3
"""
Test script to check if the phone fingerprint server is running
and if the real fingerprint app is accessible.
"""

import requests
import time
import sys

def test_server():
    """Test if the server is running and accessible"""
    print("🧪 Testing Phone Fingerprint Server...")
    print("="*50)
    
    # Test server status
    try:
        print("📡 Testing server connection...")
        response = requests.get("http://localhost:5000/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            data = response.json()
            print(f"   📊 Registered fingers: {data.get('registered_fingers', 0)}")
            print(f"   🕒 Server uptime: {data.get('uptime', 0):.1f} seconds")
        else:
            print(f"❌ Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print("   Try running: python start_server.py")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False
    
    # Test real fingerprint app
    try:
        print("\n📱 Testing real fingerprint app...")
        response = requests.get("http://localhost:5000/real_fingerprint_app.html", timeout=5)
        if response.status_code == 200:
            print("✅ Real fingerprint app is accessible!")
            print("   🌐 URL: http://localhost:5000/real_fingerprint_app.html")
        else:
            print(f"❌ Real fingerprint app returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing real fingerprint app: {e}")
        return False
    
    # Test mobile app example
    try:
        print("\n📱 Testing mobile app example...")
        response = requests.get("http://localhost:5000/mobile_app_example.html", timeout=5)
        if response.status_code == 200:
            print("✅ Mobile app example is accessible!")
            print("   🌐 URL: http://localhost:5000/mobile_app_example.html")
        else:
            print(f"❌ Mobile app example returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing mobile app example: {e}")
    
    print("\n" + "="*50)
    print("🎯 TESTING COMPLETE!")
    print("\n📱 To test on your phone:")
    print("   1. Make sure your phone and computer are on the same WiFi")
    print("   2. Find your computer's IP address")
    print("   3. Open on your phone:")
    print("      • Real fingerprint: http://YOUR_IP:5000/real_fingerprint_app.html")
    print("      • Simulated version: http://YOUR_IP:5000/mobile_app_example.html")
    print("\n💻 To test on your computer:")
    print("   • Server dashboard: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    test_server() 