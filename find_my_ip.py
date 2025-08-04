#!/usr/bin/env python3
"""
Find My IP Address
This script helps you find your computer's IP address for phone fingerprint connection.
"""

import socket
import subprocess
import platform

def get_local_ip():
    """Get the local IP address of the computer"""
    try:
        # Method 1: Connect to external server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return None

def get_ip_from_system():
    """Get IP address using system commands"""
    try:
        if platform.system() == "Windows":
            # Use ipconfig on Windows
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'IPv4 Address' in line and '192.168.' in line:
                    # Extract IP address
                    ip = line.split(':')[-1].strip()
                    if ip and ip != '192.168.56.1':  # Skip virtual adapters
                        return ip
        else:
            # Use ifconfig on Unix-like systems
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            # Parse ifconfig output for IP address
            # This is a simplified version
            pass
    except:
        pass
    return None

def main():
    print("="*50)
    print("🔍 FINDING YOUR COMPUTER'S IP ADDRESS")
    print("="*50)
    print()
    
    # Try different methods to get IP
    ip_methods = [
        ("Socket method", get_local_ip),
        ("System command", get_ip_from_system)
    ]
    
    found_ip = None
    for method_name, method_func in ip_methods:
        print(f"Trying {method_name}...")
        ip = method_func()
        if ip:
            found_ip = ip
            print(f"✅ Found IP: {ip}")
            break
        else:
            print(f"❌ {method_name} failed")
    
    if found_ip:
        print()
        print("🎯 YOUR COMPUTER'S IP ADDRESS:")
        print(f"   {found_ip}")
        print()
        print("📱 FOR PHONE FINGERPRINT CONNECTION:")
        print(f"   Server URL: http://{found_ip}:5000")
        print(f"   Mobile App: http://{found_ip}:5000/mobile_app_example.html")
        print()
        print("📋 INSTRUCTIONS:")
        print("1. Make sure your phone and computer are on the same WiFi network")
        print("2. Open the mobile app URL on your phone")
        print("3. Use your phone's fingerprint sensor")
        print("4. See all processing on your computer screen")
        print()
        
        # Show alternative IPs from your system
        print("🔍 ALTERNATIVE IP ADDRESSES (if the above doesn't work):")
        try:
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'IPv4 Address' in line and '192.168.' in line:
                    ip = line.split(':')[-1].strip()
                    if ip and ip != found_ip:
                        print(f"   {ip}")
        except:
            pass
            
    else:
        print("❌ Could not find your IP address automatically")
        print()
        print("🔧 MANUAL METHODS:")
        print("1. Open Command Prompt and type: ipconfig")
        print("2. Look for 'IPv4 Address' under your WiFi adapter")
        print("3. Use that IP address with port 5000")
        print()
        print("📱 EXAMPLE:")
        print("   If your IP is 192.168.1.100, use:")
        print("   http://192.168.1.100:5000")

if __name__ == "__main__":
    main() 