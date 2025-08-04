"""
Phone-based Fingerprint Authentication System - Main Interface
Provides a user-friendly interface for phone-based fingerprint authentication
"""

import sys
import os
from phone_scanner import PhoneFingerprintScanner
from phone_matcher import PhoneFingerprintMatcher, PhoneMatchResult
from typing import Dict

class PhoneFingerprintAuthSystem:
    def __init__(self, threshold: float = 0.75):
        self.scanner = PhoneFingerprintScanner()
        self.matcher = PhoneFingerprintMatcher(self.scanner, threshold)
        self.is_registered = len(self.scanner.scanned_fingers) > 0
        
    def show_menu(self):
        """Display the main menu"""
        print("\n" + "="*50)
        print("📱 PHONE FINGERPRINT AUTHENTICATION SYSTEM")
        print("="*50)
        print("1. 📝 Register User (Scan all 10 fingers)")
        print("2. 🔐 Authenticate User")
        print("3. ✅ Verify Registration")
        print("4. 📊 Show Statistics")
        print("5. 📱 Phone Integration Guide")
        print("6. 🗑️  Reset Registration")
        print("7. 🧪 Test Mode (Simulate Phone Scans)")
        print("8. ❌ Exit")
        print("="*50)
        
        if self.is_registered:
            print(f"✅ Status: Registered ({len(self.scanner.scanned_fingers)} fingers)")
        else:
            print("❌ Status: Not registered")
        print("="*50)
    
    def register_user(self) -> bool:
        """Register a new user by scanning all 10 fingers"""
        print("\n📝 USER REGISTRATION")
        print("="*30)
        
        if self.is_registered:
            print("⚠️  User already registered!")
            choice = input("Do you want to re-register? (y/N): ").lower()
            if choice != 'y':
                return False
            self.reset_registration()
        
        print("\n📱 PHONE INTEGRATION INSTRUCTIONS:")
        print("1. Use your phone's fingerprint sensor to scan each finger")
        print("2. Transfer the data to this system using one of these methods:")
        print("   - Web API (recommended)")
        print("   - Bluetooth transfer")
        print("   - USB connection")
        print("   - Manual file upload")
        print("3. Each finger will be processed and stored securely")
        
        print("\n🔧 For testing, we'll use simulation mode...")
        
        registered_count = 0
        for i, finger_name in enumerate(self.scanner.finger_names, 1):
            print(f"\n[{i}/10] Scanning {finger_name.replace('_', ' ').title()}...")
            
            # In real implementation, this would receive data from phone
            # For now, we'll simulate it
            result = self.scanner.simulate_phone_scan(finger_name)
            
            if result:
                registered_count += 1
                print(f"✅ {finger_name} registered successfully!")
            else:
                print(f"❌ Failed to register {finger_name}")
        
        self.is_registered = registered_count > 0
        
        if self.is_registered:
            print(f"\n🎉 Registration complete! {registered_count}/10 fingers registered.")
            return True
        else:
            print("\n❌ Registration failed. No fingers were registered.")
            return False
    
    def authenticate_user(self) -> bool:
        """Authenticate a user using their fingerprint"""
        if not self.is_registered:
            print("❌ No registered fingerprints found. Please register first.")
            return False
        
        print("\n🔐 USER AUTHENTICATION")
        print("="*30)
        
        # Show available fingers
        available_fingers = list(self.scanner.scanned_fingers)
        print("Available registered fingers:")
        for i, finger in enumerate(available_fingers, 1):
            print(f"{i}. {finger.replace('_', ' ').title()}")
        
        # Let user choose finger
        try:
            choice = int(input(f"\nSelect finger to authenticate (1-{len(available_fingers)}): ")) - 1
            if 0 <= choice < len(available_fingers):
                selected_finger = available_fingers[choice]
            else:
                print("❌ Invalid choice.")
                return False
        except ValueError:
            print("❌ Invalid input.")
            return False
        
        print(f"\n📱 Please scan your {selected_finger.replace('_', ' ')} on your phone...")
        print("(Simulating phone scan for demonstration)")
        
        # Simulate receiving data from phone
        test_data = self.scanner.simulate_phone_scan(selected_finger)
        
        if not test_data:
            print("❌ Failed to receive fingerprint data from phone.")
            return False
        
        # Authenticate
        result = self.matcher.authenticate_fingerprint(selected_finger, test_data)
        
        print(f"\n🔍 Authentication Results:")
        print(f"Match: {'✅ YES' if result.is_match else '❌ NO'}")
        print(f"Similarity Score: {result.similarity_score:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        
        if result.is_match:
            print("\n🎉 Authentication successful! Welcome back!")
        else:
            print("\n❌ Authentication failed. Please try again.")
        
        return result.is_match
    
    def verify_registration(self) -> Dict[str, bool]:
        """Verify all registered fingerprints"""
        if not self.is_registered:
            print("❌ No registered fingerprints found.")
            return {}
        
        print("\n✅ VERIFICATION MODE")
        print("="*30)
        print("Re-scanning all registered fingers to verify accuracy...")
        
        results = self.matcher.verify_all_fingers()
        
        print("\n📊 Verification Results:")
        print("-" * 40)
        
        success_count = 0
        for finger_name, result in results.items():
            status = "✅ PASS" if result.is_match else "❌ FAIL"
            print(f"{finger_name.replace('_', ' ').title()}: {status} "
                  f"(Score: {result.similarity_score:.3f})")
            if result.is_match:
                success_count += 1
        
        print(f"\nOverall: {success_count}/{len(results)} fingers verified successfully")
        
        return {finger: result.is_match for finger, result in results.items()}
    
    def show_statistics(self):
        """Display system statistics"""
        print("\n📊 SYSTEM STATISTICS")
        print("="*30)
        
        stats = self.matcher.get_matching_statistics()
        
        if "error" in stats:
            print("❌ No fingerprint data available for statistics.")
            return
        
        print(f"Total Registered Fingers: {stats['total_fingers']}")
        print(f"Average Self-Similarity: {stats['average_self_similarity']:.3f}")
        print(f"Average Cross-Similarity: {stats['average_cross_similarity']:.3f}")
        print(f"Discrimination Ratio: {stats['discrimination_ratio']:.2f}")
        print(f"Matching Threshold: {stats['threshold']:.3f}")
        
        print(f"\nRegistered Fingers:")
        for finger in stats['finger_names']:
            print(f"  - {finger.replace('_', ' ').title()}")
    
    def show_phone_integration_guide(self):
        """Show guide for phone integration"""
        print("\n📱 PHONE INTEGRATION GUIDE")
        print("="*40)
        print("This system is designed to work with your phone's fingerprint sensor.")
        print("\n🔧 INTEGRATION METHODS:")
        
        print("\n1. 🌐 WEB API (Recommended)")
        print("   - Run a Flask server on your computer")
        print("   - Use a mobile app to send fingerprint data")
        print("   - Real-time communication")
        
        print("\n2. 📡 BLUETOOTH")
        print("   - Pair your phone with computer via Bluetooth")
        print("   - Transfer fingerprint data files")
        print("   - Good for offline scenarios")
        
        print("\n3. 🔌 USB CONNECTION")
        print("   - Connect phone via USB")
        print("   - Direct data transfer")
        print("   - Fastest method")
        
        print("\n4. 📁 MANUAL FILE UPLOAD")
        print("   - Export fingerprint data from phone")
        print("   - Save as files and upload manually")
        print("   - Most flexible method")
        
        print("\n📋 REQUIRED PHONE DATA FORMAT:")
        print("- Raw sensor data (bytes)")
        print("- Sensor type (optical/capacitive/ultrasonic)")
        print("- Device information (model, OS, etc.)")
        print("- Quality metrics")
        print("- Timestamp")
        
        print("\n🔒 SECURITY FEATURES:")
        print("- Data encryption during transfer")
        print("- Secure storage on computer")
        print("- No data sent to external servers")
        print("- Local processing only")
    
    def test_mode(self):
        """Run test mode to simulate phone integration"""
        print("\n🧪 TEST MODE - SIMULATING PHONE INTEGRATION")
        print("="*50)
        
        print("This mode simulates receiving fingerprint data from a phone.")
        print("In a real implementation, this data would come from your phone's sensor.")
        
        # Test with a few fingers
        test_fingers = ["thumb_left", "index_right", "middle_left"]
        
        for finger in test_fingers:
            print(f"\n📱 Simulating phone scan for {finger.replace('_', ' ').title()}...")
            
            # Generate test data (simulating phone sensor)
            raw_data = self.scanner.generate_test_data(finger)
            
            # Simulate device info
            device_info = {
                "model": "Test Phone",
                "sensor": "optical",
                "resolution": "1920x1080",
                "os": "Android 12"
            }
            
            # Process the data
            result = self.scanner.receive_fingerprint_from_phone(
                finger, raw_data, "optical", device_info
            )
            
            if result:
                print(f"✅ Successfully processed {finger}")
                print(f"   Quality Score: {result.quality_score:.3f}")
                print(f"   Sensor Type: {result.sensor_type}")
                print(f"   Device: {result.device_info['model']}")
            else:
                print(f"❌ Failed to process {finger}")
        
        print(f"\n🎯 Test completed! {len(self.scanner.scanned_fingers)} fingers processed.")
    
    def reset_registration(self):
        """Reset all fingerprint data"""
        print("\n🗑️  RESET REGISTRATION")
        print("="*30)
        
        if not self.is_registered:
            print("❌ No registration to reset.")
            return
        
        choice = input("⚠️  This will delete ALL fingerprint data. Continue? (y/N): ").lower()
        if choice == 'y':
            self.scanner.reset_registration()
            self.is_registered = False
            print("✅ All fingerprint data has been reset.")
        else:
            print("❌ Reset cancelled.")
    
    def run(self):
        """Main application loop"""
        while True:
            self.show_menu()
            
            try:
                choice = input("\nEnter your choice (1-8): ").strip()
                
                if choice == '1':
                    self.register_user()
                elif choice == '2':
                    self.authenticate_user()
                elif choice == '3':
                    self.verify_registration()
                elif choice == '4':
                    self.show_statistics()
                elif choice == '5':
                    self.show_phone_integration_guide()
                elif choice == '6':
                    self.reset_registration()
                elif choice == '7':
                    self.test_mode()
                elif choice == '8':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-8.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    print("🚀 Starting Phone-based Fingerprint Authentication System...")
    
    # Create and run the system
    system = PhoneFingerprintAuthSystem(threshold=0.75)
    system.run()

if __name__ == "__main__":
    main() 