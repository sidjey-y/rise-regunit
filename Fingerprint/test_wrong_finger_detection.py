#!/usr/bin/env python3
"""
Test Wrong Finger Detection
Verifies that the system correctly identifies and rejects wrong fingers
"""
import sys
import time
import logging
from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem, Hand, FingerType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wrong_finger_detection():
    """Test the wrong finger detection system"""
    try:
        print("🧪 Testing Wrong Finger Detection System")
        print("=" * 50)
        
        # Initialize system
        system = ComprehensiveEnrollmentSystem()
        if not system.initialize():
            print("❌ Failed to initialize fingerprint scanner")
            return False
        
        print("✅ Scanner initialized successfully")
        print(f"📊 Current template count: {system.scanner.getTemplateCount()}")
        
        # Test 1: Try to enroll left thumb
        print("\n🧪 Test 1: Enrolling Left Thumb")
        print("-" * 30)
        print("Please place your LEFT THUMB on the sensor...")
        
        if system.guided_finger_enrollment(Hand.LEFT, FingerType.THUMB):
            print("✅ Left Thumb enrolled successfully")
        else:
            print("❌ Left Thumb enrollment failed")
            return False
        
        # Test 2: Try to enroll left thumb again (should detect duplicate)
        print("\n🧪 Test 2: Attempting to enroll Left Thumb again (should detect duplicate)")
        print("-" * 30)
        print("Please place your LEFT THUMB on the sensor again...")
        
        if system.guided_finger_enrollment(Hand.LEFT, FingerType.THUMB):
            print("❌ ERROR: System accepted duplicate finger!")
            return False
        else:
            print("✅ System correctly detected duplicate finger")
        
        # Test 3: Try to enroll right thumb (should work)
        print("\n🧪 Test 3: Enrolling Right Thumb")
        print("-" * 30)
        print("Please place your RIGHT THUMB on the sensor...")
        
        if system.guided_finger_enrollment(Hand.RIGHT, FingerType.THUMB):
            print("✅ Right Thumb enrolled successfully")
        else:
            print("❌ Right Thumb enrollment failed")
            return False
        
        # Test 4: Try to enroll left index with wrong finger (should detect wrong finger)
        print("\n🧪 Test 4: Attempting to enroll Left Index with wrong finger")
        print("-" * 30)
        print("⚠️  IMPORTANT: The system is asking for LEFT INDEX finger")
        print("   But please place your RIGHT THUMB on the sensor (wrong finger)")
        print("   The system should detect this and reject it...")
        
        if system.guided_finger_enrollment(Hand.LEFT, FingerType.INDEX):
            print("❌ ERROR: System accepted wrong finger!")
            print("   This means the wrong finger detection is not working")
            return False
        else:
            print("✅ System correctly detected wrong finger")
        
        # Test 5: Now enroll left index with correct finger
        print("\n🧪 Test 5: Enrolling Left Index with correct finger")
        print("-" * 30)
        print("Please place your LEFT INDEX finger on the sensor...")
        
        if system.guided_finger_enrollment(Hand.LEFT, FingerType.INDEX):
            print("✅ Left Index enrolled successfully")
        else:
            print("❌ Left Index enrollment failed")
            return False
        
        # Summary
        print("\n🎉 WRONG FINGER DETECTION TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("✅ All tests passed:")
        print("   • Left Thumb enrolled successfully")
        print("   • Duplicate finger detection working")
        print("   • Right Thumb enrolled successfully")
        print("   • Wrong finger detection working")
        print("   • Left Index enrolled successfully")
        
        print(f"\n📊 Final template count: {system.scanner.getTemplateCount()}")
        print("💾 Saving test data...")
        system.save_enrollment_data()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        if 'system' in locals():
            system.cleanup()

def main():
    """Main function"""
    print("Starting wrong finger detection test...")
    success = test_wrong_finger_detection()
    
    if success:
        print("\n🎉 All tests passed! Wrong finger detection is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Wrong finger detection needs improvement.")
        sys.exit(1)

if __name__ == "__main__":
    main()
