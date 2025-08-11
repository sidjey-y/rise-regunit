#!/usr/bin/env python3
"""
Interactive Fingerprint Enrollment
Step-by-step guidance for enrolling all 10 fingers
"""

import sys
import time
import logging
from datetime import datetime
from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem, Hand, FingerType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_welcome():
    """Print welcome message and instructions"""
    print("=" * 70)
    print("🔐 COMPREHENSIVE FINGERPRINT ENROLLMENT SYSTEM")
    print("=" * 70)
    print()
    print("This system will guide you through enrolling all 10 fingers:")
    print("  • Left Hand:  Thumb, Index, Middle, Ring, Little")
    print("  • Right Hand: Thumb, Index, Middle, Ring, Little")
    print()
    print("📋 IMPORTANT REQUIREMENTS:")
    print("  • All 10 fingers must be enrolled in one session (within 5 minutes)")
    print("  • Each finger will be scanned and verified")
    print("  • The system prevents duplicate finger enrollment")
    print("  • Raw image data and features will be extracted")
    print()
    print("⏰ Session timeout: 5 minutes")
    print("=" * 70)
    print()

def print_finger_guide(hand: Hand, finger_type: FingerType, step: int):
    """Print step-by-step finger guide"""
    print(f"\n📝 STEP {step}: {hand.value.upper()} {finger_type.value.upper()}")
    print("-" * 50)
    
    # Provide specific guidance for each finger
    if finger_type == FingerType.THUMB:
        print("👆 Place your THUMB on the sensor")
        print("   • Position: Pad of thumb facing down")
        print("   • Apply gentle, even pressure")
        print("   ⚠️  IMPORTANT: Make sure this is your THUMB, not index or other finger")
    elif finger_type == FingerType.INDEX:
        print("👆 Place your INDEX finger on the sensor")
        print("   • Position: Pad of index finger facing down")
        print("   • Keep finger straight and centered")
        print("   ⚠️  IMPORTANT: Make sure this is your INDEX finger, not thumb or middle")
    elif finger_type == FingerType.MIDDLE:
        print("👆 Place your MIDDLE finger on the sensor")
        print("   • Position: Pad of middle finger facing down")
        print("   • Ensure good contact with sensor")
        print("   ⚠️  IMPORTANT: Make sure this is your MIDDLE finger, not index or ring")
    elif finger_type == FingerType.RING:
        print("👆 Place your RING finger on the sensor")
        print("   • Position: Pad of ring finger facing down")
        print("   • Maintain steady pressure")
        print("   ⚠️  IMPORTANT: Make sure this is your RING finger, not middle or little")
    elif finger_type == FingerType.LITTLE:
        print("👆 Place your LITTLE finger on the sensor")
        print("   • Position: Pad of little finger facing down")
        print("   • Keep finger steady and centered")
        print("   ⚠️  IMPORTANT: Make sure this is your LITTLE finger, not ring or other")
    
    print(f"\n🔄 System will verify this is the correct {finger_type.value.title()}")
    print("   If wrong finger detected, you'll be asked to scan again")
    print("\n⏳ Waiting for finger...")

def print_progress(enrolled: int, total: int):
    """Print enrollment progress"""
    percentage = (enrolled / total) * 100
    bar_length = 30
    filled_length = int(bar_length * enrolled // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"\n📊 PROGRESS: {enrolled}/{total} fingers ({percentage:.1f}%)")
    print(f"[{bar}]")
    
    if enrolled == total:
        print("🎉 ALL FINGERS ENROLLED!")
    else:
        remaining = total - enrolled
        print(f"⏳ {remaining} finger(s) remaining...")

def print_session_status(system: ComprehensiveEnrollmentSystem):
    """Print current session status"""
    if system.is_session_active():
        elapsed = time.time() - system.session_start_time
        remaining = system.session_timeout - elapsed
        print(f"\n⏰ Session Time: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
        
        if remaining < 60:
            print("⚠️  WARNING: Less than 1 minute remaining!")
        elif remaining < 120:
            print("⚠️  WARNING: Less than 2 minutes remaining!")
    else:
        print("\n❌ SESSION EXPIRED! Must complete within 5 minutes.")

def interactive_enrollment():
    """Run interactive enrollment process"""
    try:
        print_welcome()
        
        # Initialize system
        system = ComprehensiveEnrollmentSystem()
        
        if not system.initialize():
            print("❌ Failed to initialize fingerprint scanner")
            return False
        
        # Start session
        system.start_enrollment_session()
        
        print("🚀 Enrollment session ready!")
        print("⏰ Session will start when you're ready")
        print("📋 You will need to scan all 10 fingers:")
        print("   • Left Hand: Thumb, Index, Middle, Ring, Pinky")
        print("   • Right Hand: Thumb, Index, Middle, Ring, Pinky")
        print("\n" + "="*60)
        input("Press Enter when you're ready to start scanning...")
        print()
        
        print("🚀 Starting enrollment session...")
        print("⏰ Session started at:", datetime.now().strftime("%H:%M:%S"))
        print()
        
        # Enroll each finger
        for step, (hand, finger_type) in enumerate(system.required_fingers, 1):
            print_finger_guide(hand, finger_type, step)
            
            # Check session timeout
            if not system.is_session_active():
                print("\n❌ SESSION TIMEOUT EXCEEDED!")
                print("Enrollment must be completed within 5 minutes.")
                print("Please restart the enrollment process.")
                return False
            
            # Enroll finger
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                print(f"🔄 Attempt {retry_count + 1}/{max_retries}")
                
                # Use guided enrollment for better finger verification
                if system.guided_finger_enrollment(hand, finger_type):
                    print(f"✅ {hand.value.title()} {finger_type.value.title()} enrolled successfully!")
                    break
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"❌ Enrollment failed. Retry {retry_count}/{max_retries}")
                        
                        # Get detailed feedback about what went wrong
                        feedback = system.get_enrollment_feedback(hand, finger_type)
                        print(f"💡 Feedback: {feedback}")
                        
                        # Analyze scan issues
                        issues = system.analyze_scan_issues(hand, finger_type)
                        if issues:
                            print("🔍 Scan Analysis:")
                            for issue in issues:
                                print(f"   • {issue}")
                        
                        print(f"\n⚠️  Please ensure you're scanning the correct finger:")
                        print(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                        print("   Clean the sensor and try again...")
                        time.sleep(2)
                        print_finger_guide(hand, finger_type, step)
                    else:
                        print(f"❌ Failed to enroll {hand.value.title()} {finger_type.value.title()} after {max_retries} attempts")
                        
                        # Final analysis
                        issues = system.analyze_scan_issues(hand, finger_type)
                        if issues:
                            print("🔍 Final Analysis:")
                            for issue in issues:
                                print(f"   • {issue}")
                        
                        print("Please restart the enrollment process.")
                        return False
            
            # Show progress
            enrolled, total = system.get_enrollment_progress()
            print_progress(enrolled, total)
            
            # Show session status
            print_session_status(system)
            
            # Prompt user to continue to next finger
            if step < len(system.required_fingers):
                print(f"\n🎯 {hand.value.title()} {finger_type.value.title()} successfully enrolled!")
                print("📸 Raw image data and minutiae points captured and saved.")
                print(f"⏭️  Ready to scan next finger: {system.required_fingers[step][0].value.title()} {system.required_fingers[step][1].value.title()}")
                print("\n" + "="*60)
                input("Press Enter to continue to the next finger...")
                print()
        
        # Prompt user before verification phase
        print("\n🎯 All 10 fingers have been enrolled successfully!")
        print("📸 Raw image data and minutiae points captured for all fingers.")
        print("\n" + "="*60)
        input("Press Enter to start the verification phase...")
        print()
        
        # Final verification
        print("\n🔍 VERIFICATION PHASE")
        print("=" * 50)
        print("Now we'll verify all enrolled fingers...")
        print()
        
        all_verified = True
        for hand, finger_type in system.required_fingers:
            print(f"🔍 Verifying {hand.value.title()} {finger_type.value.title()}...")
            print("Place the same finger on the sensor...")
            
            if system.verify_finger(hand, finger_type):
                print(f"✅ {hand.value.title()} {finger_type.value.title()} verified!")
            else:
                print(f"❌ {hand.value.title()} {finger_type.value.title()} verification failed!")
                all_verified = False
            
            print()
        
        if all_verified:
            print("🎉 ALL FINGERS SUCCESSFULLY ENROLLED AND VERIFIED!")
            print("=" * 50)
            
            # Save data
            system.save_enrollment_data()
            
            # Show summary
            enrolled, total = system.get_enrollment_progress()
            print(f"📊 Final Summary: {enrolled}/{total} fingers enrolled")
            print(f"⏱️  Total session time: {time.time() - system.session_start_time:.1f} seconds")
            print(f"💾 Data saved to: enrollment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            print("\n" + "="*60)
            input("Press Enter to complete the enrollment session...")
            print()
            
            return True
        else:
            print("❌ Some fingers failed verification")
            print("Please review and re-enroll if necessary.")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Enrollment interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    finally:
        if 'system' in locals():
            system.cleanup()

def main():
    """Main function"""
    print("Starting interactive fingerprint enrollment...")
    
    success = interactive_enrollment()
    
    if success:
        print("\n🎉 Enrollment completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Enrollment failed or was interrupted")
        sys.exit(1)

if __name__ == "__main__":
    main()
