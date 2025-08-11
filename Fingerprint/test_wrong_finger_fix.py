#!/usr/bin/env python3
"""
Test Wrong Finger Detection Fix
Verifies that the system correctly rejects wrong fingers
"""
import sys
import time
import logging
from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem, Hand, FingerType

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wrong_finger_detection():
    """Test the wrong finger detection system"""
    try:
        print("🧪 Testing Wrong Finger Detection Fix")
        print("=" * 50)
        print("This test will guide you through enrolling different fingers")
        print("and testing wrong finger detection.")
        print()
        
        # Initialize system
        system = ComprehensiveEnrollmentSystem()
        if not system.initialize():
            print("❌ Failed to initialize fingerprint scanner")
            return False
        
        print("✅ Scanner initialized successfully")
        print(f"📊 Current template count: {system.scanner.getTemplateCount()}")
        
        # Test 1: Enroll left thumb first
        print("\n🧪 Test 1: Enrolling Left Thumb")
        print("-" * 30)
        print("Please place your LEFT THUMB on the sensor...")
        input("Press Enter when you're ready to scan your LEFT THUMB...")
        
        if system.guided_finger_enrollment(Hand.LEFT, FingerType.THUMB):
            print("✅ Left Thumb enrolled successfully")
        else:
            print("❌ Left Thumb enrollment failed")
            return False
        
        # Test 2: Try to enroll left index with right index (should detect wrong finger)
        print("\n🧪 Test 2: Attempting to enroll Left Index with Right Index (should detect wrong finger)")
        print("-" * 30)
        print("⚠️  IMPORTANT: The system is asking for LEFT INDEX finger")
        print("   But please place your RIGHT INDEX finger on the sensor (wrong finger)")
        print("   The system should detect this and reject it...")
        print()
        print("💡 Instructions:")
        print("   1. Remove your LEFT THUMB from the sensor")
        print("   2. Place your RIGHT INDEX finger on the sensor")
        print("   3. The system should reject this wrong finger")
        input("Press Enter when you're ready to scan your RIGHT INDEX (wrong finger)...")
        
        if system.guided_finger_enrollment(Hand.LEFT, FingerType.INDEX):
            print("❌ ERROR: System accepted wrong finger!")
            print("   This means the wrong finger detection is still not working")
            return False
        else:
            print("✅ System correctly detected wrong finger")
        
        # Test 3: Now enroll left index with correct finger
        print("\n🧪 Test 3: Enrolling Left Index with correct finger")
        print("-" * 30)
        print("Now let's enroll the correct finger - your LEFT INDEX")
        print()
        print("💡 Instructions:")
        print("   1. Remove your RIGHT INDEX from the sensor")
        print("   2. Place your LEFT INDEX finger on the sensor")
        print("   3. This should be accepted as the correct finger")
        input("Press Enter when you're ready to scan your LEFT INDEX (correct finger)...")
        
        if system.guided_finger_enrollment(Hand.LEFT, FingerType.INDEX):
            print("✅ Left Index enrolled successfully")
        else:
            print("❌ Left Index enrollment failed")
            return False
        
        # Test 4: Try to enroll right thumb with left thumb (should detect wrong finger)
        print("\n🧪 Test 4: Attempting to enroll Right Thumb with Left Thumb (should detect wrong finger)")
        print("-" * 30)
        print("⚠️  IMPORTANT: The system is asking for RIGHT THUMB finger")
        print("   But please place your LEFT THUMB finger on the sensor (wrong finger)")
        print("   The system should detect this and reject it...")
        print()
        print("💡 Instructions:")
        print("   1. Remove your LEFT INDEX from the sensor")
        print("   2. Place your LEFT THUMB finger on the sensor (wrong hand)")
        print("   3. The system should reject this wrong hand")
        input("Press Enter when you're ready to scan your LEFT THUMB (wrong hand)...")
        
        if system.guided_finger_enrollment(Hand.RIGHT, FingerType.THUMB):
            print("❌ ERROR: System accepted wrong finger!")
            print("   This means the wrong finger detection is still not working")
            return False
        else:
            print("✅ System correctly detected wrong finger")
        
        # Test 5: Now enroll right thumb with correct finger
        print("\n🧪 Test 5: Enrolling Right Thumb with correct finger")
        print("-" * 30)
        print("Now let's enroll the correct finger - your RIGHT THUMB")
        print()
        print("💡 Instructions:")
        print("   1. Remove your LEFT THUMB from the sensor")
        print("   2. Place your RIGHT THUMB finger on the sensor")
        print("   3. This should be accepted as the correct finger")
        input("Press Enter when you're ready to scan your RIGHT THUMB (correct finger)...")
        
        if system.guided_finger_enrollment(Hand.RIGHT, FingerType.THUMB):
            print("✅ Right Thumb enrolled successfully")
        else:
            print("❌ Right Thumb enrollment failed")
            return False
        
        # Summary
        print("\n🎉 WRONG FINGER DETECTION TEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("✅ All tests passed:")
        print("   • Left Thumb enrolled successfully")
        print("   • Wrong finger detection working (right index rejected for left index)")
        print("   • Left Index enrolled successfully")
        print("   • Wrong finger detection working (left thumb rejected for right thumb)")
        print("   • Right Thumb enrolled successfully")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
    finally:
        try:
            system.cleanup()
            print("🧹 System cleanup completed")
        except:
            pass

def main():
    """Main function"""
    print("Starting wrong finger detection fix test...")
    success = test_wrong_finger_detection()
    
    if success:
        print("\n🎉 All tests passed! Wrong finger detection is now working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Wrong finger detection still needs improvement.")
        sys.exit(1)

if __name__ == "__main__":
    main()
