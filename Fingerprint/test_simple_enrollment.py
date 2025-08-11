#!/usr/bin/env python3
"""
Simple 10-Finger Enrollment Test
This script tests the basic enrollment functionality without fake finger validation.
"""

import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('simple_enrollment_test.log')
        ]
    )

def test_simple_enrollment():
    """Test the simple enrollment system"""
    try:
        from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem, Hand, FingerType
        
        print("Simple 10-Finger Enrollment Test")
        print("=" * 50)
        print("This test will:")
        print("1. Initialize the scanner")
        print("2. Enroll all 10 fingers")
        print("3. Save data in JSON and YAML formats")
        print("4. Extract minutiae points for each finger")
        print()
        
        # Initialize the system
        print("Initializing enrollment system...")
        enrollment_system = ComprehensiveEnrollmentSystem()
        
        if not enrollment_system.initialize():
            print("Failed to initialize system")
            return False
        
        print("System initialized successfully")
        
        # Test scanner connection
        print("\nTesting scanner connection...")
        if not enrollment_system.test_scanner_connection():
            print("Scanner connection test failed!")
            return False
        
        print("Scanner connection test passed!")
        
        # Start enrollment session
        print("\nStarting enrollment session...")
        enrollment_system.start_enrollment_session()
        
        # Enroll each finger
        required_fingers = [
            (Hand.LEFT, FingerType.THUMB),
            (Hand.LEFT, FingerType.INDEX),
            (Hand.LEFT, FingerType.MIDDLE),
            (Hand.LEFT, FingerType.RING),
            (Hand.LEFT, FingerType.LITTLE),
            (Hand.RIGHT, FingerType.THUMB),
            (Hand.RIGHT, FingerType.INDEX),
            (Hand.RIGHT, FingerType.MIDDLE),
            (Hand.RIGHT, FingerType.RING),
            (Hand.RIGHT, FingerType.LITTLE)
        ]
        
        enrolled_count = 0
        
        for hand, finger_type in required_fingers:
            print(f"\nEnrolling: {hand.value.title()} {finger_type.value.title()}")
            print("-" * 40)
            
            # Ask for confirmation
            response = input(f"Ready to scan {hand.value.title()} {finger_type.value.title()}? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print(f"Skipping {hand.value.title()} {finger_type.value.title()}")
                continue
            
            # Try to enroll the finger
            if enrollment_system.enroll_finger(hand, finger_type):
                enrolled_count += 1
                print(f"✅ Successfully enrolled {hand.value.title()} {finger_type.value.title()}")
                
                # Show progress
                print(f"Progress: {enrolled_count}/10 fingers enrolled")
            else:
                print(f"❌ Failed to enroll {hand.value.title()} {finger_type.value.title()}")
                print("You can continue with other fingers or retry this one")
                
                retry = input("Retry this finger? (yes/no): ").strip().lower()
                if retry in ['yes', 'y']:
                    # Try again
                    if enrollment_system.enroll_finger(hand, finger_type):
                        enrolled_count += 1
                        print(f"✅ Successfully enrolled {hand.value.title()} {finger_type.value.title()} on retry")
                    else:
                        print(f"❌ Failed to enroll {hand.value.title()} {finger_type.value.title()} on retry")
        
        # Save enrollment data
        print(f"\nSaving enrollment data...")
        enrollment_system.save_enrollment_data()
        
        # Show final summary
        print(f"\nEnrollment completed!")
        print(f"Total fingers enrolled: {enrolled_count}/10")
        
        if enrolled_count == 10:
            print("🎉 All 10 fingers enrolled successfully!")
        else:
            print(f"⚠️  Only {enrolled_count}/10 fingers enrolled")
            print("You can run the enrollment again to complete the remaining fingers")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required files are in the same directory.")
        return False
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

def main():
    """Main test function"""
    setup_logging()
    
    print("Simple 10-Finger Enrollment Test")
    print("=" * 50)
    print("This test focuses on what actually works:")
    print("1. Scanner initialization and connection")
    print("2. Finger enrollment with user guidance")
    print("3. Data storage in JSON/YAML formats")
    print("4. Minutiae point extraction")
    print()
    
    success = test_simple_enrollment()
    
    if success:
        print("\nTest completed successfully!")
        print("Check the generated JSON and YAML files for your enrollment data.")
    else:
        print("\nTest failed. Check the logs for more details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
