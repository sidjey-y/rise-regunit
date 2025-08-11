#!/usr/bin/env python3
"""
Enhanced Fingerprint Enrollment System
This system enrolls exactly 10 fingers for 1 user with duplication detection.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main enrollment function"""
    print("🔐 ENHANCED FINGERPRINT ENROLLMENT SYSTEM")
    print("=" * 60)
    print("This system will enroll exactly 10 fingers for 1 user")
    print("Features:")
    print("  ✅ 10 fingers per user (left/right: thumb, index, middle, ring, little)")
    print("  ✅ Duplication detection within the same user")
    print("  ✅ 3 attempts per finger if needed")
    print("  ✅ User prompts before each finger scan")
    print("  ✅ Minutiae extraction and storage")
    print("  ✅ JSON and YAML output formats")
    print("  ✅ Real-time progress tracking")
    print("=" * 60)
    
    try:
        from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem
        
        # Get user ID
        user_id = input("\nEnter User ID (e.g., 'user001', 'john_doe'): ").strip()
        if not user_id:
            print("❌ User ID is required!")
            return False
        
        print(f"\n🎯 Enrolling user: {user_id}")
        print("=" * 40)
        
        # Initialize the enrollment system
        enrollment_system = ComprehensiveEnrollmentSystem()
        
        # Run the complete enrollment
        success = enrollment_system.run_complete_enrollment(user_id)
        
        if success:
            print("\n🎉 ENROLLMENT COMPLETED SUCCESSFULLY!")
            print(f"✅ User {user_id} now has all 10 fingers enrolled")
            print("📁 Data saved to JSON and YAML files")
            print("🔒 Scanner templates updated")
        else:
            print("\n❌ ENROLLMENT FAILED!")
            print("Please check the error messages above")
        
        return success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required files are in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Enrollment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
