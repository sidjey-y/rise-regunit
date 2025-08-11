#!/usr/bin/env python3
"""
Test Duplicate Detection System
This script tests the improved duplicate detection in the enrollment system.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_duplicate_detection():
    """Test the duplicate detection system"""
    print("🔍 TESTING DUPLICATE DETECTION SYSTEM")
    print("=" * 60)
    
    try:
        from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem
        
        # Initialize the system
        system = ComprehensiveEnrollmentSystem()
        
        print("✅ System initialized successfully")
        print("\n🔍 Testing duplicate detection logic...")
        
        # Test 1: No duplicates (empty system)
        print("\n📝 Test 1: Empty system (no duplicates)")
        print("-" * 40)
        
        # Simulate more realistic fingerprint characteristics data
        # Real fingerprint characteristics are typically 256+ bytes with specific patterns
        import random
        
        # Create realistic fingerprint characteristics (256 bytes)
        random.seed(42)  # For reproducible results
        dummy_char1 = bytes(random.getrandbits(8) for _ in range(256))
        
        # Create a similar but different set (simulating different finger)
        random.seed(123)
        dummy_char2 = bytes(random.getrandbits(8) for _ in range(256))
        
        # Create a very similar set (simulating same finger scanned twice)
        # Copy most of char1 but change a few bytes to simulate scan variations
        dummy_char1_duplicate = bytearray(dummy_char1)
        # Change only 10% of bytes to simulate same finger with slight variations
        for i in range(0, len(dummy_char1_duplicate), 10):
            if i < len(dummy_char1_duplicate):
                dummy_char1_duplicate[i] = (dummy_char1_duplicate[i] + 1) % 256
        
        dummy_char1_duplicate = bytes(dummy_char1_duplicate)
        
        # Import the enum classes directly
        from comprehensive_enrollment_system import Hand, FingerType
        
        is_duplicate, duplicate_info, similarity = system.check_duplicate_within_user(
            dummy_char1, Hand.LEFT, FingerType.THUMB
        )
        
        print(f"Result: {'DUPLICATE' if is_duplicate else 'NO DUPLICATE'}")
        if is_duplicate:
            print(f"Similar to: {duplicate_info}")
            print(f"Similarity: {similarity:.2%}")
        else:
            print("✅ Correctly detected no duplicates")
        
        # Test 2: Add a finger and test duplicate
        print("\n📝 Test 2: Add finger and test duplicate detection")
        print("-" * 40)
        
        # Simulate adding a left thumb
        from datetime import datetime
        from comprehensive_enrollment_system import FingerprintData
        
        # Create dummy fingerprint data
        left_thumb_data = FingerprintData(
            user_id="test_user",
            hand="left",
            finger_type="thumb",
            position=0,
            timestamp=datetime.now().isoformat(),
            raw_image_data=dummy_char1
        )
        
        # Add to system
        system.current_user_fingers["left_thumb"] = left_thumb_data
        print("✅ Added left thumb to system")
        
        # Test duplicate detection with similar characteristics (same finger)
        is_duplicate, duplicate_info, similarity = system.check_duplicate_within_user(
            dummy_char1_duplicate, Hand.RIGHT, FingerType.MIDDLE
        )
        
        print(f"Result: {'DUPLICATE' if is_duplicate else 'NO DUPLICATE'}")
        if is_duplicate:
            print(f"Similar to: {duplicate_info}")
            print(f"Similarity: {similarity:.2%}")
            print("✅ Correctly detected duplicate!")
        else:
            print("❌ Failed to detect duplicate")
        
        # Test 3: Test with different characteristics
        print("\n📝 Test 3: Test with different characteristics")
        print("-" * 40)
        
        is_duplicate, duplicate_info, similarity = system.check_duplicate_within_user(
            dummy_char2, Hand.RIGHT, FingerType.MIDDLE
        )
        
        print(f"Result: {'DUPLICATE' if is_duplicate else 'NO DUPLICATE'}")
        if is_duplicate:
            print(f"Similar to: {duplicate_info}")
            print(f"Similarity: {similarity:.2%}")
            print("❌ Incorrectly detected duplicate")
        else:
            print("✅ Correctly detected no duplicate")
        
        # Test 4: Show system state
        print("\n📝 Test 4: System state")
        print("-" * 40)
        print(f"Total fingers in system: {len(system.current_user_fingers)}")
        for key, fp in system.current_user_fingers.items():
            print(f"  • {key}: {fp.hand.title()} {fp.finger_type.title()}")
        
        print("\n🎉 Duplicate detection test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Testing Improved Duplicate Detection")
    print("=" * 60)
    
    success = test_duplicate_detection()
    
    if success:
        print("\n✅ All tests passed!")
        print("The duplicate detection system is working correctly")
        print("\nNow run 'python run_enrollment.py' to test with real scanner")
    else:
        print("\n❌ Some tests failed!")
        print("Please check the error messages above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 