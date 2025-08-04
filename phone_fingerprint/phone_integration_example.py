

import json
import base64
import time
from phone_scanner import PhoneFingerprintScanner
from phone_matcher import PhoneFingerprintMatcher

class PhoneIntegrationExample:
    def __init__(self):
        self.scanner = PhoneFingerprintScanner()
        self.matcher = PhoneFingerprintMatcher(self.scanner, threshold=0.75)
        
    def receive_from_phone_api(self, phone_data_json: str):

        try:
            # Parse the JSON data from phone
            phone_data = json.loads(phone_data_json)
            
            print("📱 Received data from phone:")
            print(f"   Finger: {phone_data['finger_id']}")
            print(f"   Sensor: {phone_data['sensor_type']}")
            print(f"   Device: {phone_data['device_info']['model']}")
            print(f"   Quality: {phone_data['quality_score']:.3f}")
            
            # Decode the raw sensor data
            raw_data = base64.b64decode(phone_data['raw_data'])
            
            # Process the fingerprint data
            result = self.scanner.receive_fingerprint_from_phone(
                phone_data['finger_id'],
                raw_data,
                phone_data['sensor_type'],
                phone_data['device_info']
            )
            
            if result:
                print("✅ Successfully processed phone fingerprint data!")
                return result
            else:
                print("❌ Failed to process phone fingerprint data")
                return None
                
        except Exception as e:
            print(f"❌ Error processing phone data: {e}")
            return None
    
    def simulate_phone_capacitive_scan(self, finger_name: str, phone_model: str = "Samsung Galaxy"):
        """
        Simulate receiving data from a phone's capacitive fingerprint sensor
        In real usage, this data would come from your actual phone
        """
        print(f"\n📱 Simulating capacitive fingerprint scan from {phone_model}...")
        
        # Simulate phone sensor data (in real case, this comes from your phone)
        sensor_data = self.scanner.generate_test_data(finger_name)
        
        # Create phone data structure (this is what your phone would send)
        phone_data = {
            "finger_id": finger_name,
            "sensor_type": "capacitive",
            "raw_data": base64.b64encode(sensor_data).decode('utf-8'),
            "device_info": {
                "model": phone_model,
                "sensor": "capacitive",
                "os": "Android 12",
                "sensor_resolution": "192x192"
            },
            "quality_score": 0.87,
            "timestamp": time.time()
        }
        
        # Convert to JSON (as your phone would send)
        phone_data_json = json.dumps(phone_data)
        
        # Process the data
        return self.receive_from_phone_api(phone_data_json)
    
    def authenticate_with_phone_scan(self, finger_name: str):
        """
        Authenticate using a new scan from your phone
        """
        if not self.scanner.scanned_fingers:
            print("❌ No registered fingerprints found. Please register first.")
            return False
        
        print(f"\n🔐 Authenticating with phone scan for {finger_name}...")
        
        # Simulate new scan from phone
        new_scan = self.simulate_phone_capacitive_scan(finger_name)
        
        if not new_scan:
            print("❌ Failed to receive scan from phone")
            return False
        
        # Authenticate against registered fingerprints
        result = self.matcher.authenticate_fingerprint(finger_name, new_scan)
        
        print(f"\n🔍 Authentication Results:")
        print(f"   Match: {'✅ YES' if result.is_match else '❌ NO'}")
        print(f"   Similarity: {result.similarity_score:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        
        return result.is_match
    
    def demo_phone_integration(self):
        """
        Demonstrate the complete phone integration workflow
        """
        print("="*60)
        print("📱 PHONE FINGERPRINT INTEGRATION DEMO")
        print("="*60)
        print()
        print("This demo shows how to:")
        print("1. Receive fingerprint data from your phone's capacitive sensor")
        print("2. Process the data on your computer")
        print("3. Store and authenticate fingerprints")
        print()
        
        # Step 1: Register fingerprints using phone
        print("📝 STEP 1: Register fingerprints using phone sensor")
        print("-" * 50)
        
        fingers_to_register = ["thumb_left", "index_right", "middle_left"]
        
        for finger in fingers_to_register:
            print(f"\n📱 Registering {finger} using phone sensor...")
            result = self.simulate_phone_capacitive_scan(finger)
            
            if result:
                print(f"✅ {finger} registered successfully!")
            else:
                print(f"❌ Failed to register {finger}")
        
        # Step 2: Authenticate using phone
        print(f"\n🔐 STEP 2: Authenticate using phone sensor")
        print("-" * 50)
        
        for finger in fingers_to_register:
            if finger in self.scanner.scanned_fingers:
                success = self.authenticate_with_phone_scan(finger)
                if success:
                    print(f"🎉 Authentication successful for {finger}!")
                else:
                    print(f"❌ Authentication failed for {finger}")
        
        # Step 3: Show statistics
        print(f"\n📊 STEP 3: System Statistics")
        print("-" * 50)
        
        stats = self.matcher.get_matching_statistics()
        if "error" not in stats:
            print(f"Registered Fingers: {stats['total_fingers']}")
            print(f"Average Self-Similarity: {stats['average_self_similarity']:.3f}")
            print(f"Discrimination Ratio: {stats['discrimination_ratio']:.2f}")
        
        print(f"\n🎯 Demo completed! You can now:")
        print("- Use your real phone's capacitive sensor")
        print("- Send data to this system using any integration method")
        print("- See all processing and results on your computer")

def main():
    """Main function to run the phone integration demo"""
    demo = PhoneIntegrationExample()
    demo.demo_phone_integration()

if __name__ == "__main__":
    main() 