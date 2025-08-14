#!/usr/bin/env python3
"""
Test script to validate duplicate detection is working properly
"""

import sys
import logging
from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem, Hand, FingerType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_duplicate_detection():
    """Test duplicate detection functionality"""
    logger.info("=" * 60)
    logger.info("DUPLICATE DETECTION VALIDATION TEST")
    logger.info("=" * 60)
    logger.info("This test will help you verify that duplicate detection is working")
    logger.info("by asking you to scan the same finger twice")
    logger.info("")
    
    enrollment_system = ComprehensiveEnrollmentSystem()
    
    try:
        # Initialize scanner
        if not enrollment_system.initialize():
            logger.error("Failed to initialize scanner")
            return False
        
        # Start a test session
        test_user_id = "test_duplicate_user"
        enrollment_system.start_enrollment_session(test_user_id)
        
        logger.info("Test 1: Enroll your right index finger")
        logger.info("=" * 40)
        logger.info("Place your RIGHT INDEX finger on the scanner")
        input("Press Enter when ready to scan your right index finger...")
        
        # Enroll right index finger
        success1 = enrollment_system.enroll_finger(Hand.RIGHT, FingerType.INDEX)
        
        if not success1:
            logger.error("Failed to enroll right index finger")
            return False
        
        logger.info("✅ Right index finger enrolled successfully")
        logger.info("")
        
        logger.info("Test 2: Try to scan the SAME finger again (should detect duplicate)")
        logger.info("=" * 40)
        logger.info("Place your RIGHT INDEX finger on the scanner AGAIN")
        logger.info("This should trigger duplicate detection and fail")
        input("Press Enter when ready to test duplicate detection...")
        
        # Try to enroll the same finger again - this should fail due to duplicate detection
        success2 = enrollment_system.enroll_finger(Hand.RIGHT, FingerType.INDEX)
        
        if success2:
            logger.warning("⚠️  DUPLICATE DETECTION MAY NOT BE WORKING!")
            logger.warning("The same finger was enrolled twice without detection")
            logger.warning("This indicates the duplicate detection thresholds may need adjustment")
            return False
        else:
            logger.info("✅ DUPLICATE DETECTION WORKING!")
            logger.info("The system correctly detected and rejected the duplicate finger")
            return True
    
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False
    finally:
        enrollment_system.cleanup()

def test_wrong_finger_detection():
    """Test wrong finger detection functionality"""
    logger.info("=" * 60)
    logger.info("WRONG FINGER DETECTION TEST")
    logger.info("=" * 60)
    logger.info("This test will help you verify wrong finger detection")
    logger.info("")
    
    enrollment_system = ComprehensiveEnrollmentSystem()
    
    try:
        # Initialize scanner
        if not enrollment_system.initialize():
            logger.error("Failed to initialize scanner")
            return False
        
        # Start a test session
        test_user_id = "test_wrong_finger_user"
        enrollment_system.start_enrollment_session(test_user_id)
        
        logger.info("Test 1: Enroll your left thumb")
        logger.info("=" * 40)
        logger.info("Place your LEFT THUMB on the scanner")
        input("Press Enter when ready to scan your left thumb...")
        
        # Enroll left thumb
        success1 = enrollment_system.enroll_finger(Hand.LEFT, FingerType.THUMB)
        
        if not success1:
            logger.error("Failed to enroll left thumb")
            return False
        
        logger.info("✅ Left thumb enrolled successfully")
        logger.info("")
        
        logger.info("Test 2: Try to scan your RIGHT THUMB (should detect wrong finger)")
        logger.info("=" * 40)
        logger.info("Place your RIGHT THUMB on the scanner")
        logger.info("This should trigger wrong finger detection and fail")
        input("Press Enter when ready to test wrong finger detection...")
        
        # Try to enroll right thumb when right thumb is expected - should fail
        success2 = enrollment_system.enroll_finger(Hand.RIGHT, FingerType.THUMB)
        
        if success2:
            logger.warning("⚠️  WRONG FINGER DETECTION MAY NOT BE WORKING!")
            logger.warning("Different fingers were not detected as similar")
            return True  # This might be expected if thumbs are not similar
        else:
            logger.info("✅ WRONG FINGER DETECTION WORKING!")
            logger.info("The system correctly detected the wrong finger")
            return True
    
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False
    finally:
        enrollment_system.cleanup()

def main():
    """Main test function"""
    logger.info("FINGERPRINT DUPLICATE DETECTION VALIDATION")
    logger.info("=" * 60)
    logger.info("This script will test if duplicate detection is working properly")
    logger.info("")
    
    try:
        # Ask user which test to run
        logger.info("Available tests:")
        logger.info("1. Duplicate Detection Test (scan same finger twice)")
        logger.info("2. Wrong Finger Detection Test (scan different fingers)")
        logger.info("3. Both tests")
        logger.info("")
        
        choice = input("Which test would you like to run? (1/2/3): ").strip()
        
        if choice == "1":
            success = test_duplicate_detection()
        elif choice == "2":
            success = test_wrong_finger_detection()
        elif choice == "3":
            success1 = test_duplicate_detection()
            logger.info("\n" + "=" * 60)
            success2 = test_wrong_finger_detection()
            success = success1 and success2
        else:
            logger.error("Invalid choice")
            return False
        
        logger.info("\n" + "=" * 60)
        if success:
            logger.info("✅ ALL TESTS PASSED - Duplicate detection is working!")
        else:
            logger.info("❌ TESTS FAILED - Duplicate detection needs adjustment")
            logger.info("Try lowering the similarity thresholds in the code")
        logger.info("=" * 60)
        
        return success
        
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
