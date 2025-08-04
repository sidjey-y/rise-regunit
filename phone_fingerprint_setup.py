#!/usr/bin/env python3
"""
Phone Fingerprint Setup Guide
This script helps you set up phone-based fingerprint authentication
using your phone's capacitive fingerprint sensor.
"""

import sys
import os

def main():
    print("="*60)
    print("📱 PHONE FINGERPRINT AUTHENTICATION SETUP")
    print("="*60)
    print()
    print("This system allows you to:")
    print("1. Use your phone's capacitive fingerprint sensor")
    print("2. See all processing and results on your computer")
    print("3. Store and authenticate fingerprints locally")
    print()
    
    print("🔧 SETUP INSTRUCTIONS:")
    print("="*40)
    print()
    print("1. INSTALL DEPENDENCIES:")
    print("   pip install numpy opencv-python Pillow scipy scikit-learn")
    print()
    print("2. RUN THE SYSTEM:")
    print("   python phone_fingerprint/phone_main.py")
    print()
    print("3. PHONE INTEGRATION METHODS:")
    print("   - Web API (recommended for real-time)")
    print("   - Bluetooth file transfer")
    print("   - USB connection")
    print("   - Manual file upload")
    print()
    print("4. FOR TESTING (without real phone):")
    print("   - Select option 7: Test Mode")
    print("   - System will simulate phone scans")
    print()
    
    print("📱 PHONE DATA FORMAT NEEDED:")
    print("="*40)
    print("Your phone should send data in this format:")
    print("""
{
  "finger_id": "thumb_left",
  "sensor_type": "capacitive",
  "raw_data": "base64_encoded_sensor_data",
  "device_info": {
    "model": "Your Phone Model",
    "sensor": "capacitive",
    "os": "Android/iOS version"
  },
  "quality_score": 0.85,
  "timestamp": 1234567890.123
}
""")
    
    print("🚀 QUICK START:")
    print("="*40)
    print("1. Run: python phone_fingerprint/phone_main.py")
    print("2. Select option 7 for test mode")
    print("3. Follow the prompts to simulate phone scans")
    print("4. See real-time processing on your computer")
    print()
    
    print("💡 TIPS FOR PHONE INTEGRATION:")
    print("="*40)
    print("- Use your phone's fingerprint sensor to capture data")
    print("- Transfer data to computer using your preferred method")
    print("- All processing happens on your computer")
    print("- Results are displayed in real-time")
    print("- Data is stored securely locally")
    print()
    
    choice = input("Press Enter to try running the system now, or 'q' to quit: ")
    if choice.lower() != 'q':
        try:
            print("\n🚀 Starting Phone Fingerprint System...")
            print("="*50)
            
            # Add the phone_fingerprint directory to Python path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'phone_fingerprint'))
            
            # Import and run the main system
            from phone_main import main as phone_main
            phone_main()
            
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Please make sure you're in the correct directory and dependencies are installed.")
            print("Try: pip install -r phone_fingerprint/requirements.txt")
        except Exception as e:
            print(f"❌ Error running system: {e}")
            print("Please check the setup instructions above.")

if __name__ == "__main__":
    main() 